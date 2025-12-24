# Copyright 2025 Xiaomi Corporation.
import gradio as gr
import torch
import os
import tempfile
import argparse
from pathlib import Path
from src.mimo_audio.mimo_audio import MimoAudio


class TTSGenerator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def generate(self, text, instruct, output_audio_path):
        path = Path(output_audio_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text_output = self.model.tts_sft(text, output_audio_path, instruct)
        return text_output

class AudioUnderstandingGenerator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def generate(self, input_speech, input_text, thinking=False):
        text = self.model.audio_understanding_sft(input_speech, input_text, thinking=thinking)
        return text

class SpokenDialogueGenerator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def generate(self, input_speech, output_audio_path, system_prompt="You are MiMo-Audio, a friendly AI assistant and your response needs to be concise.", prompt_speech=None, add_history=False):
        
        path = Path(output_audio_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text_response = self.model.spoken_dialogue_sft(input_speech, output_audio_path, system_prompt=system_prompt, prompt_speech=prompt_speech, add_history=add_history)
        return text_response
    
    def clear_history(self):
        self.model.clear_history()

class Speech2TextDialogueGenerator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def generate(self, input_speech, thinking=False, add_history=False):
        text = self.model.speech2text_dialogue_sft(input_speech, thinking=thinking, add_history=add_history)
        return text
    
    def clear_history(self):
        self.model.clear_history()


class TextDialogueGenerator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def generate(self, input_text, thinking=False, add_history=False):
        text = self.model.text_dialogue_sft(input_text, thinking=thinking, add_history=add_history)
        return text
    
    def clear_history(self):
        self.model.clear_history()


class MultiModalSpeechInterface:
    def __init__(self):
        self.model = None
        self.tts_generator = None
        self.audio_understanding_generator = None
        self.spoken_dialogue_generator = None
        self.speech2text_dialogue_generator = None
        self.text_dialogue_generator = None
        
        self.device = None
        self.model_initialized = False
        
    def initialize_model(self, model_path=None, tokenizer_path=None):

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if model_path is None:
                model_path = "./models/MiMo-Audio-7B-Instruct"
            if tokenizer_path is None:
                tokenizer_path = "./models/MiMo-Audio-Tokenizer"
            

            print(f"Model path: {model_path}")
            print(f"Tokenizer path: {tokenizer_path}")
            
            self.model = MimoAudio(model_path, tokenizer_path)
            self.tts_generator = TTSGenerator(self.model, self.device)
            self.audio_understanding_generator = AudioUnderstandingGenerator(self.model, self.device)
            self.spoken_dialogue_generator = SpokenDialogueGenerator(self.model, self.device)
            self.speech2text_dialogue_generator = Speech2TextDialogueGenerator(self.model, self.device)
            self.text_dialogue_generator = TextDialogueGenerator(self.model, self.device)
            
            
            self.model_initialized = True
            print("Model loaded successfully!")
            return "✅ Model loaded successfully!"
            
        except Exception as e:
            error_msg = f"❌ Model loading failed: {str(e)}"
            print(error_msg)
            return error_msg

    def generate_tts_audio(self, input_text, instruct="", use_instruct=False):
        if not self.model_initialized:
            return None, "❌ Error: Model not initialized, please load the model first"
        
        if not input_text.strip():
            return None, "❌ Error: Please input text content"
        
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            
            if not (use_instruct and instruct.strip()):
                instruct = None
                       
            print(f"Generating TTS audio: {input_text}")

            text_channel = self.tts_generator.generate(input_text, instruct, output_path)
            status_msg = f"✅ TTS audio generated successfully!\n📝 Input text: {input_text}"
            if use_instruct and instruct is not None and instruct.strip():
                status_msg += f"\n🎭 Style description: {instruct}"
            status_msg += f"\n🎵 Output text channel: {text_channel}"

            return output_path, status_msg, gr.update(value=output_path, visible=True)
            
        except Exception as e:
            error_msg = f"❌ Error generating TTS audio: {str(e)}"
            print(error_msg)
            return None, error_msg, gr.update(visible=False)


    def generate_audio_understanding_response(self, input_audio, input_text, thinking=False):
        if not self.model_initialized:
            return "", "❌ Error: Model not initialized, please load the model first"
        
        if input_audio is None and not input_text.strip():
            return "", "❌ Error: Please provide either audio input or text question"
        
        if input_audio is None:
            return "", "❌ Error: Please upload an audio file for Audio Understanding task"
        
        if not input_text.strip():
            return "", "❌ Error: Please input your question"
        
        try:
            print(f"Performing Audio Understanding task:")
            print(f"Audio input: {input_audio}")
            print(f"Text question: {input_text}")
            
            
            audio_understanding_response = self.audio_understanding_generator.generate(input_audio, input_text.strip(), thinking=thinking)
            
            status_msg = f"✅ Audio Understanding task completed successfully!\n🎵 Audio input: {os.path.basename(input_audio)}\n❓ Question: {input_text}\n💬 Answer: {audio_understanding_response}"
            
            return audio_understanding_response, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error performing Audio Understanding task: {str(e)}"
            print(error_msg)
            return "", error_msg

    def generate_spoken_dialogue_response(self, input_audio, system_prompt=None, prompt_speech=None, add_history=False):
        if not self.model_initialized:
            return "", "❌ Error: Model not initialized, please load the model first"
        
        if input_audio is None:
            return "", "❌ Error: Please upload an audio file"
        
        try:
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_audio_path = tmp_file.name
            
            print(f"Performing spoken dialogue task:")
            print(f"Audio input: {input_audio}")
            print(f"Output path: {output_audio_path}")
            
            
            dialogue_response = self.spoken_dialogue_generator.generate(input_audio, output_audio_path, system_prompt=system_prompt, prompt_speech=prompt_speech, add_history=add_history)
            
            status_msg = f"✅ Spoken dialogue task completed successfully!\n🎵 Audio input: {os.path.basename(input_audio)}\n💬 Response: {dialogue_response[:300]}..."
            
            return output_audio_path, dialogue_response, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error performing spoken dialogue task: {str(e)}"
            print(error_msg)
            return None, None, error_msg


    def generate_speech2text_dialogue_response(self, input_audio, thinking=False, add_history=False):
        if not self.model_initialized:
            return "", "❌ Error: Model not initialized, please load the model first"
        
        if input_audio is None:
            return "", "❌ Error: Please upload an audio file for S2T Dialogue task"
        
        
        try:
            print(f"Performing S2T Dialogue task:")
            print(f"Audio input: {input_audio}")
            
            
            s2t_response = self.speech2text_dialogue_generator.generate(input_audio, thinking=thinking, add_history=add_history)
            
            status_msg = f"✅ S2T dialogue task completed successfully!\n🎵 Audio input: {os.path.basename(input_audio)}\n❓💬 Answer: {s2t_response}"
            
            return s2t_response, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error performing QA task: {str(e)}"
            print(error_msg)
            return "", error_msg

    def generate_text_dialogue_response(self, input_text, thinking=False, add_history=False):
        if not self.model_initialized:
            return "", "❌ Error: Model not initialized, please load the model first"
        
        if not input_text or not input_text.strip():
            return "", "❌ Error: Please input your text"
        
        try:
            print(f"Performing Text Dialogue task:")
            print(f"Text input: {input_text}")
            print(f"Thinking mode: {thinking}")
            print(f"Add history: {add_history}")
            
            
            t2t_response = self.text_dialogue_generator.generate(input_text.strip(), thinking=thinking, add_history=add_history)
            
            status_msg = f"✅ T2T dialogue task completed successfully!\n💬 Input: {input_text}"
            if thinking:
                status_msg += f"\n🧠 Thinking mode: Enabled"
            status_msg += f"\n💬 Answer: {t2t_response}"
            
            return t2t_response, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error performing T2T dialogue task: {str(e)}"
            print(error_msg)
            return "", error_msg

    def clear_spoken_dialogue_history(self):
        if not self.model_initialized:
            return None, "", "❌ Error: Model not initialized, please load the model first"
        
        try:
            self.spoken_dialogue_generator.clear_history()
            return None, "", "✅ Spoken dialogue history cleared successfully!"
        except Exception as e:
            error_msg = f"❌ Error clearing spoken dialogue history: {str(e)}"
            print(error_msg)
            return None, "", error_msg
    
    def clear_speech2text_dialogue_history(self):
        if not self.model_initialized:
            return "", "❌ Error: Model not initialized, please load the model first"
        
        try:
            self.speech2text_dialogue_generator.clear_history()
            return "", "✅ Speech-to-text dialogue history cleared successfully!"
        except Exception as e:
            error_msg = f"❌ Error clearing S2T dialogue history: {str(e)}"
            print(error_msg)
            return "", error_msg

    def clear_text_dialogue_history(self):
        if not self.model_initialized:
            return "", "❌ Error: Model not initialized, please load the model first"
        
        try:
            self.text_dialogue_generator.clear_history()
            return "", "✅ Text dialogue history cleared successfully!"
        except Exception as e:
            error_msg = f"❌ Error clearing T2T dialogue history: {str(e)}"
            print(error_msg)
            return "", error_msg



    def create_interface(self):
        
        with gr.Blocks(title="MiMo-Audio Multimodal Speech Processing System", theme=gr.themes.Soft()) as iface:
            gr.Markdown("# 🎵 MiMo-Audio Multimodal Speech Processing System")
            gr.Markdown("Supports audio understanding, text-to-speech, spoken dialogue, speech-to-text dialogue and text-to-text dialogue")
            
            with gr.Tabs():
                
                with gr.TabItem("⚙️ Model Configuration"):
                    gr.Markdown("### 📋 Model initialization configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            
                            model_path = gr.Textbox(
                                label="Model path",
                                placeholder="Leave blank to use default path: ./models/MiMo-Audio-7B-Instruct",
                                lines=3
                            )
                            
                            tokenizer_path = gr.Textbox(
                                label="Tokenizer path",
                                placeholder="Leave blank to use default path: ./models/MiMo-Audio-Tokenizer",
                                lines=3
                            )
                            
                            init_btn = gr.Button("🔄 Initialize model", variant="primary", size="lg")
                            
                        with gr.Column():
                            init_status = gr.Textbox(
                                label="Initialization status",
                                interactive=False,
                                lines=6,
                                placeholder="Click the initialize model button to start..."
                            )
                            
                            
                            gr.Markdown("### 💻 System information")
                            device_info = gr.Textbox(
                                label="Device information",
                                value=f"GPU available: {'Yes' if torch.cuda.is_available() else 'No'}",
                                interactive=False
                            )
                
                
                with gr.TabItem("🔊 Audio Understanding"):
                    gr.Markdown("### 🎯 Audio Understanding")
                    
                    with gr.Row():
                        with gr.Column():
                            audio_understanding_input_audio = gr.Audio(
                                label="Upload Audio File",
                                type="filepath",
                                interactive=True,
                            )
                            
                            audio_understanding_input_text = gr.Textbox(
                                label="Input Question",
                                placeholder="Please input your question...",
                                lines=3,
                            )
                            
                            audio_understanding_thinking = gr.Checkbox(
                                label="Enable Thinking Mode",
                                value=False,
                                info="Enable thinking mode, AI will perform a deeper analysis and thinking"
                            )
                            
                            audio_understanding_generate_btn = gr.Button("🤖 Start Audio Understanding", variant="primary", size="lg")
                            
                            
                            
                        with gr.Column():
                            audio_understanding_output_text = gr.Textbox(
                                label="Answer Result",
                                lines=8,
                                interactive=False,
                                placeholder="AI's answer will be displayed here...",
                                elem_id="audio_understanding_output_text"
                            )
                            
                            audio_understanding_status = gr.Textbox(
                                label="Processing Status",
                                lines=6,
                                interactive=False,
                                placeholder="Processing status information will be displayed here..."
                            )

                            with gr.Row():
                                audio_understanding_copy_btn = gr.Button("📋 Copy Answer", size="sm")
                                audio_understanding_clear_btn = gr.Button("🗑️ Clear Result", size="sm")
                    
                    gr.Markdown("### 🌟 Audio Understanding Examples")
                    audio_understanding_examples = gr.Examples(
                        examples=[
                            [None, "这段音频的主要内容是什么？"],
                            [None, "说话者的情感状态如何？"],
                            [None, "音频中提到了哪些关键信息？"],
                            [None, "Please summarize the main points of this conversation."],
                            [None, "What viewpoint does the speaker want to express?"]
                        ],
                        inputs=[audio_understanding_input_audio, audio_understanding_input_text],
                        label="Click the example to automatically fill the question"
                    )
                    

                
                
                with gr.TabItem("🎵 Text-to-Speech"):
                    gr.Markdown("### 🎵 Text-to-Speech")
                    
                    with gr.Row():
                        with gr.Column():
                            
                            tts_input_text = gr.Textbox(
                                label="Input Text",
                                placeholder="Please input the text you want to convert to speech...",
                                lines=4,
                                max_lines=8
                            )
                            
                            tts_instruct = gr.Textbox(
                                label="Style Description (Optional)",
                                placeholder="Please input the style description (optional)...",
                                lines=3,
                                max_lines=5
                            )
                            
                            tts_use_instruct = gr.Checkbox(
                                label="Use Style Description",
                                value=True,
                                info="Enable to use InstructTTS for style-controlled speech generation"
                            )
                            
                            tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                            
                        with gr.Column():
                            
                            tts_output_audio = gr.Audio(
                                label="Generated Speech",
                                type="filepath"
                            )
                            
                            tts_status = gr.Textbox(
                                label="Generation Status",
                                lines=6,
                                interactive=False
                            )

                            
                            tts_download_btn = gr.DownloadButton(
                                label="Download Generated Audio",
                                visible=False
                            )
                    
                
                
                
                with gr.TabItem("🎤 Spoken Dialogue"):
                    gr.Markdown("### 🎯 Spoken Dialogue")
                    
                    with gr.Row():
                        with gr.Column():
                            
                            dialogue_input_audio = gr.Audio(
                                label="Upload User Speech",
                                type="filepath",
                                interactive=True
                            )
                            system_prompt = gr.Textbox(
                                label="System Prompt (Optional)",
                                placeholder="e.g.: You are MiMo-Audio, a friendly AI assistant and your response needs to be concise.",
                                lines=1
                            )
                            prompt_speech = gr.Audio(
                                label="Prompt Speech (Optional, MiMo-Audio speaks with the same timbre as your prompt.)",
                                type="filepath",
                                interactive=True
                            )
                            spoken_dialogue_add_history = gr.Checkbox(
                                label="Enable History Record",
                                value=True,
                                info="Enable to remember the previous dialogue context"
                            )
                            
                            with gr.Row():
                                dialogue_generate_btn = gr.Button("💬 Start Dialogue", variant="primary", size="lg")
                            
                            with gr.Row():
                                dialogue_clear_history_btn = gr.Button("🗑️ Clear Dialogue History", size="sm", variant="secondary")
                            

                            

                            
                        with gr.Column():
                            
                            dialogue_output_audio = gr.Audio(
                                label="Output Audio",
                                type="filepath"
                            )
                            dialogue_output_text = gr.Textbox(
                                label="Dialogue Response",
                                lines=5,
                                interactive=False,
                            )
                            dialogue_status = gr.Textbox(
                                label="Dialogue Status",
                                lines=5,
                                interactive=False,
                            )
                    
                    
                    

                
                with gr.TabItem("💬 S2T Dialogue"):
                    gr.Markdown("### 🎯 S2T Dialogue")
                    
                    with gr.Row():
                        with gr.Column():
                            
                            s2t_dialogue_input_audio = gr.Audio(
                                label="Upload User Speech",
                                type="filepath",
                                interactive=True
                            )
                            
                            
                            s2t_dialogue_add_history = gr.Checkbox(
                                label="Enable History Record",
                                value=True,
                                info="Enable to remember the previous dialogue context"
                            )
                            
                            s2t_dialogue_thinking = gr.Checkbox(
                                label="Enable Thinking Mode (think mode)",
                                value=False,
                                info="Enable to perform a deeper analysis and reasoning"
                            )
                            
                            with gr.Row():
                                s2t_dialogue_generate_btn = gr.Button("🎧 Start S2T Dialogue", variant="primary", size="lg")
                            
                            with gr.Row():
                                s2t_dialogue_clear_history_btn = gr.Button("🗑️ Clear Dialogue History", size="sm", variant="secondary")
                            
                            
                        with gr.Column():
                            
                            s2t_dialogue_output_text = gr.Textbox(
                                label="Dialogue Response",
                                lines=8,
                                interactive=False,
                                placeholder="AI's dialogue response will be displayed here..."
                            )
                            
                            s2t_dialogue_status = gr.Textbox(
                                label="Dialogue Status",
                                lines=5,
                                interactive=False,
                                placeholder="Dialogue status information will be displayed here..."
                            )
                    

                
                with gr.TabItem("📝 T2T Dialogue"):
                    gr.Markdown("### 🎯 T2T Dialogue")
                    
                    with gr.Row():
                        with gr.Column():
                            
                            t2t_dialogue_input_text = gr.Textbox(
                                label="Input Dialogue Content",
                                placeholder="Please input the text content you want to dialogue...",
                                lines=4,
                                max_lines=8
                            )
                            
                            t2t_dialogue_add_history = gr.Checkbox(
                                label="Enable History Record",
                                value=True,
                                info="Enable to remember the previous dialogue context"
                            )
                            
                            t2t_dialogue_thinking = gr.Checkbox(
                                label="Enable Thinking Mode (Thinking)",
                                value=False,
                                info="Enable thinking mode, AI will perform a deeper analysis and thinking"
                            )
                            
                            with gr.Row():
                                t2t_dialogue_generate_btn = gr.Button("💬 Start T2T Dialogue", variant="primary", size="lg")
                            
                            with gr.Row():
                                t2t_dialogue_clear_history_btn = gr.Button("🗑️ Clear Dialogue History", size="sm", variant="secondary")
                            
                            
                            
                        with gr.Column():
                            t2t_dialogue_output_text = gr.Textbox(
                                label="Dialogue Response",
                                lines=8,
                                interactive=False,
                                placeholder="AI's dialogue response will be displayed here..."
                            )
                            
                            t2t_dialogue_status = gr.Textbox(
                                label="Dialogue Status",
                                lines=5,
                                interactive=False,
                                placeholder="Dialogue status information will be displayed here..."
                            )
                    
                    gr.Markdown("### 🌟 T2T Dialogue Examples")
                    t2t_dialogue_examples = gr.Examples(
                        examples=[
                            ["Hello, how are you?"],
                            ["I want to know the history of the development of artificial intelligence"],
                            ["Please recommend some good movies"],
                            ["Can you help me explain the basic concepts of quantum physics?"],
                            ["I'm learning programming recently, any suggestions?"]
                        ],
                        inputs=[t2t_dialogue_input_text],
                        label="Click the example to automatically fill the dialogue content"
                    )
                    
            
            
            def copy_text_to_clipboard(text):
                return text
            
            def clear_audio_understanding_results():
                return "", "🗑️ Audio Understanding Result Cleared"
                
            
            init_btn.click(
                fn=lambda path, tok_path: self.initialize_model(path or None, tok_path or None),
                inputs=[model_path, tokenizer_path],
                outputs=[init_status]
            )
            
            
            audio_understanding_generate_btn.click(
                fn=self.generate_audio_understanding_response,
                inputs=[audio_understanding_input_audio, audio_understanding_input_text, audio_understanding_thinking],
                outputs=[audio_understanding_output_text, audio_understanding_status]
            )

            audio_understanding_copy_btn.click(
                fn=None,
                inputs=[audio_understanding_output_text],
                js="(text) => {navigator.clipboard.writeText(text); alert('Copied to clipboard!')}"
            )
            
            tts_generate_btn.click(
                fn=self.generate_tts_audio,
                inputs=[tts_input_text, tts_instruct, tts_use_instruct],
                outputs=[tts_output_audio, tts_status, tts_download_btn]
            )
            
            dialogue_generate_btn.click(
                fn=self.generate_spoken_dialogue_response,
                inputs=[dialogue_input_audio, system_prompt, prompt_speech, spoken_dialogue_add_history],
                outputs=[dialogue_output_audio, dialogue_output_text, dialogue_status]
            )
        
            
            
            dialogue_clear_history_btn.click(
                fn=self.clear_spoken_dialogue_history,
                outputs=[dialogue_output_audio, dialogue_output_text, dialogue_status]
            )
            
            
            s2t_dialogue_generate_btn.click(
                fn=self.generate_speech2text_dialogue_response,
                inputs=[s2t_dialogue_input_audio, s2t_dialogue_thinking, s2t_dialogue_add_history],
                outputs=[s2t_dialogue_output_text, s2t_dialogue_status]
            )
            
            
            
            s2t_dialogue_clear_history_btn.click(
                fn=self.clear_speech2text_dialogue_history,
                outputs=[s2t_dialogue_output_text, s2t_dialogue_status]
            )
            
            
            t2t_dialogue_generate_btn.click(
                fn=self.generate_text_dialogue_response,
                inputs=[t2t_dialogue_input_text, t2t_dialogue_thinking, t2t_dialogue_add_history],
                outputs=[t2t_dialogue_output_text, t2t_dialogue_status]
            )
            
            
            t2t_dialogue_clear_history_btn.click(
                fn=self.clear_text_dialogue_history,
                outputs=[t2t_dialogue_output_text, t2t_dialogue_status]
            )
            
            
            
            
            audio_understanding_clear_btn.click(
                fn=clear_audio_understanding_results,
                outputs=[audio_understanding_output_text, audio_understanding_status]
            )
            
            
            
           
            
            
            tts_input_text.submit(
                fn=self.generate_tts_audio,
                inputs=[tts_input_text, tts_instruct, tts_use_instruct],
                outputs=[tts_output_audio, tts_status, tts_download_btn]
            )
            
            
            audio_understanding_input_text.submit(
                fn=self.generate_audio_understanding_response,
                inputs=[audio_understanding_input_audio, audio_understanding_input_text, audio_understanding_thinking],
                outputs=[audio_understanding_output_text, audio_understanding_status]
            )
            
            t2t_dialogue_input_text.submit(
                fn=self.generate_text_dialogue_response,
                inputs=[t2t_dialogue_input_text, t2t_dialogue_thinking, t2t_dialogue_add_history],
                outputs=[t2t_dialogue_output_text, t2t_dialogue_status]
            )

        
        return iface

def main():
    parser = argparse.ArgumentParser(description="MiMo-Audio")
    parser.add_argument("--host", default="0.0.0.0", help="Server Address")
    parser.add_argument("--port", type=int, default=7897, help="Port")
    parser.add_argument("--share", action="store_true", help="Create Public Link")
    parser.add_argument("--debug", action="store_true", help="Debug Mode")
    
    args = parser.parse_args()
    
    
    
    print("🚀 Launch MiMo-Audio...")
    
    
    speech_interface = MultiModalSpeechInterface()
    
    
    
    print("🎨 Create Gradio Interface...")
    iface = speech_interface.create_interface()
    
    
    print(f"🌐 Launch Service - Address: {args.host}:{args.port}")
    
    iface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )

if __name__ == "__main__":
    main()