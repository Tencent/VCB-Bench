// @ts-nocheck
/**
 * 音频队列播放器 - 优化网络抖动处理
 * 
 * 设计思路：
 * 1. 所有收到的音频帧按顺序入队
 * 2. 使用较大的初始缓冲区减少网络抖动影响
 * 3. 不丢弃任何音频帧，确保完整播放
 * 4. Buffer underrun 时快速恢复
 */

function asMs(samples) {
  return (samples * 1000 / sampleRate).toFixed(1);
}

function asSamples(mili) {
  return Math.round(mili * sampleRate / 1000);
}

class MoshiProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    console.log("Audio Queue Processor initialized", currentFrame, sampleRate);
    
    // ============ 缓冲区配置 ============
    const frameSize = asSamples(40);  // 每帧 80ms
    
    // 初始缓冲：收集多少数据后开始播放（越大越平滑，但延迟越高）
    // 建议 240-320ms (3-4帧)，可以容忍 2-3 个帧的网络抖动
    this.initialBufferSamples = 12 * frameSize;  // 240ms
    
    // 启动后的额外等待时间
    this.startupDelaySamples = asSamples(40);  // 40ms
    
    // Buffer underrun 后的最小恢复缓冲
    this.minRecoveryBuffer = asSamples(40);  // 80ms
    
    // 状态初始化
    this.initState();

    // 消息处理
    this.port.onmessage = (event) => {
      if (event.data.type === "reset") {
        console.log("[AudioProcessor] Reset state");
        this.initState();
        return;
      }
      
      // 将音频帧加入队列
      const frame = event.data.frame;
      this.audioQueue.push(frame);
      this.totalQueuedSamples += frame.length;
      
      // 检查是否可以开始播放
      if (!this.isPlaying && this.totalQueuedSamples >= this.initialBufferSamples) {
        this.startPlayback();
      }
      
      // 日志（前20个包）
      if (this.packetCount < 20) {
        console.log(
          this.timestamp(),
          `[Queue] Packet #${this.packetCount++}`,
          `queued: ${asMs(this.totalQueuedSamples)}ms`,
          `frame: ${asMs(frame.length)}ms`
        );
      }
      
      // 发送状态给主线程
      this.port.postMessage({
        totalAudioPlayed: this.totalAudioPlayed,
        actualAudioPlayed: this.actualAudioPlayed,
        delay: event.data.micDuration - this.playbackPosition,
        minDelay: this.minDelay,
        maxDelay: this.maxDelay,
        queueSize: this.totalQueuedSamples,
      });
    };
  }

  initState() {
    // 音频队列 - 存储所有待播放的音频帧
    this.audioQueue = [];
    this.totalQueuedSamples = 0;
    
    // 当前帧的播放偏移
    this.currentFrameOffset = 0;
    
    // 播放状态
    this.isPlaying = false;
    this.startupCountdown = 0;  // 启动倒计时
    this.isFirstOutput = false;
    
    // 播放位置（已播放的样本数）
    this.playbackPosition = 0;
    
    // 统计指标
    this.totalAudioPlayed = 0;    // 总播放时长（含静音）
    this.actualAudioPlayed = 0;   // 实际播放时长
    this.maxDelay = 0;
    this.minDelay = 9999;
    
    // 调试
    this.packetCount = 0;
    this.underrunCount = 0;
    
    console.log("[AudioProcessor] State initialized");
  }

  timestamp() {
    return `[${Date.now() % 10000}]`;
  }

  startPlayback() {
    this.isPlaying = true;
    this.startupCountdown = this.startupDelaySamples;
    this.isFirstOutput = true;
    console.log(
      this.timestamp(),
      `[Playback] Starting with ${asMs(this.totalQueuedSamples)}ms buffered`
    );
  }

  // 从队列中获取音频数据
  getAudioFromQueue(requestedSamples) {
    const outputBuffer = new Float32Array(requestedSamples);
    let outputOffset = 0;
    
    while (outputOffset < requestedSamples && this.audioQueue.length > 0) {
      const currentFrame = this.audioQueue[0];
      const availableInFrame = currentFrame.length - this.currentFrameOffset;
      const samplesToTake = Math.min(availableInFrame, requestedSamples - outputOffset);
      
      // 复制数据
      outputBuffer.set(
        currentFrame.subarray(
          this.currentFrameOffset,
          this.currentFrameOffset + samplesToTake
        ),
        outputOffset
      );
      
      outputOffset += samplesToTake;
      this.currentFrameOffset += samplesToTake;
      this.totalQueuedSamples -= samplesToTake;
      
      // 当前帧播放完毕，移到下一帧
      if (this.currentFrameOffset >= currentFrame.length) {
        this.audioQueue.shift();
        this.currentFrameOffset = 0;
      }
    }
    
    return { buffer: outputBuffer, actualSamples: outputOffset };
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0][0];
    const outputLength = output.length;
    
    // 更新延迟统计
    const currentDelay = this.totalQueuedSamples / sampleRate;
    if (this.isPlaying && this.totalQueuedSamples > 0) {
      this.maxDelay = Math.max(this.maxDelay, currentDelay);
      this.minDelay = Math.min(this.minDelay, currentDelay);
    }
    
    // 还未开始播放 - 输出静音
    if (!this.isPlaying) {
      // output 默认已经是 0
      return true;
    }
    
    // 启动倒计时 - 额外等待一小段时间让缓冲更充足
    if (this.startupCountdown > 0) {
      this.startupCountdown -= outputLength;
      return true;
    }
    
    // 从队列获取音频
    const { buffer, actualSamples } = this.getAudioFromQueue(outputLength);
    
    // 复制到输出
    if (actualSamples > 0) {
      output.set(buffer.subarray(0, actualSamples), 0);
      
      // 首次输出 - 淡入效果
      if (this.isFirstOutput) {
        this.isFirstOutput = false;
        console.log(
          this.timestamp(),
          `[Playback] First output, queue: ${asMs(this.totalQueuedSamples)}ms`
        );
        // 淡入
        for (let i = 0; i < Math.min(actualSamples, 256); i++) {
          output[i] *= i / 256;
        }
      }
    }
    
    // Buffer underrun - 队列空了
    if (actualSamples < outputLength) {
      this.underrunCount++;
      console.log(
        this.timestamp(),
        `[Underrun #${this.underrunCount}] Missing ${outputLength - actualSamples} samples,`,
        `queue: ${asMs(this.totalQueuedSamples)}ms`
      );
      
      // 淡出已播放的部分，避免爆音
      if (actualSamples > 0) {
        const fadeLength = Math.min(actualSamples, 128);
        for (let i = 0; i < fadeLength; i++) {
          output[actualSamples - 1 - i] *= i / fadeLength;
        }
      }
      
      // 如果队列完全清空，重置播放状态，下一轮会重新等待初始缓冲区
      if (this.totalQueuedSamples === 0 && this.audioQueue.length === 0) {
        this.isPlaying = false;
        this.currentFrameOffset = 0;
        this.isFirstOutput = false;
        console.log(this.timestamp(), "[Playback] Queue empty, reset for next turn - will wait for initial buffer again");
      } else if (this.totalQueuedSamples < this.minRecoveryBuffer) {
        // 等待最小恢复缓冲后再继续播放
        // 不完全重置，只是暂停等待缓冲
        this.startupCountdown = this.minRecoveryBuffer - this.totalQueuedSamples;
      }
    }
    
    // 更新统计
    this.totalAudioPlayed += outputLength / sampleRate;
    this.actualAudioPlayed += actualSamples / sampleRate;
    this.playbackPosition += actualSamples / sampleRate;
    
    return true;
  }
}

registerProcessor("moshi-processor", MoshiProcessor);
