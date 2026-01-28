import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation, ConversationMode } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";


function getFloatFromStorage(val: string | null) {
  return (val == null) ? undefined : parseFloat(val)
}

function getIntFromStorage(val: string | null) {
  return (val == null) ? undefined : parseInt(val)
}

export const Queue: FC = () => {
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [shouldConnect, setShouldConnect] = useState<boolean>(false);
  const [connectionMode, setConnectionMode] = useState<ConversationMode | null>(null);
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const modelParams = useModelParams({
    textTemperature: getFloatFromStorage(localStorage.getItem("textTemperature")),
    textTopk: getIntFromStorage(localStorage.getItem("textTopk")),
    audioTemperature: getFloatFromStorage(localStorage.getItem("audioTemperature")),
    audioTopk: getIntFromStorage(localStorage.getItem("audioTopk")),
    padMult: getFloatFromStorage(localStorage.getItem("padMult")),
    repetitionPenalty: getFloatFromStorage(localStorage.getItem("repetitionPenalty")),
    repetitionPenaltyContext: getIntFromStorage(localStorage.getItem("repetitionPenaltyContext")),
    imageResolution: getIntFromStorage(localStorage.getItem("imageResolution"))
  });

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  // enable eruda in development
  useEffect(() => {
    if (env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if (env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch (e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if (!audioContext.current) {
      audioContext.current = new AudioContext();
    }
    if (worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const onConnect = useCallback(async (mode: ConversationMode) => {
    await startProcessor();
    const hasAccess = await getMicrophoneAccess();
    if (hasAccess) {
      setConnectionMode(mode);
      setShouldConnect(true);
    }
  }, [setShouldConnect, startProcessor, getMicrophoneAccess, setConnectionMode]);

  if (hasMicrophoneAccess && audioContext.current && worklet.current && connectionMode) {
    // workerAddr 直接使用模式名，Vite 代理会根据路径转发
    const workerAddr = overrideWorkerAddr ?? connectionMode;
    return (
      <Conversation
        workerAddr={workerAddr}
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
        mode={connectionMode}
        systemPrompt={systemPrompt}
        {...modelParams}
      />
    );
  }

  return (
    <div className="text-white text-center h-screen w-screen p-4 flex flex-col items-center ">
      <div>
        <h1 className="text-4xl" style={{ letterSpacing: "5px" }}>S2S-Demo</h1>
        <div className="pt-8 text-sm flex justify-center items-center flex-col ">
          <div className="presentation text-center">
            <p>你好 欢迎使用</p>
          </div>
        </div>
      </div>
      <div className="flex flex-grow justify-center items-center flex-col presentation">
        <div className="w-full max-w-xl mb-6 px-4">
          <label className="block text-left mb-2 text-sm opacity-80">System Prompt (可选)</label>
          <textarea
            className="w-full h-32 p-3 bg-black border-2 border-white text-white rounded-none resize-none focus:outline-none focus:border-blue-400"
            placeholder="You are asked to generate both text and speech tokens at the same time. 你的名字是小云。你是一位来自杭州的温柔友善的女孩，声音甜美，举止亲切。你的回复语气自然友好，力求沟通简洁明了。你的回复简短，通常只有一到三句话，避免使用正式的称谓和重复的短语。你能用恰当的声音回复，遵循用户的指示，并能共情他们的情绪。你能用恰当的方言回复，会说四川话和粤语。"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
        </div>
        <div className="flex gap-4">
          <Button onClick={async () => await onConnect('simplex')}>
            <span className="flex flex-col items-center">
              <span>开始连接</span>
            </span>
          </Button>
        </div>
      </div>
      <div className="flex flex-grow justify-center items-center flex-col">
        {showMicrophoneAccessMessage &&
          <p className="text-center">Please enable your microphone before proceeding</p>
        }
      </div>
    </div >
  )
};
