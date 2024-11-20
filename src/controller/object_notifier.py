from google.cloud import texttospeech
from dotenv import load_dotenv
from playsound import playsound
import os

class TextToSpeechNotifier:
    def __init__(self):
        load_dotenv()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.client = texttospeech.TextToSpeechClient()
        self.output_file = "output.mp3"
        self.speech_speed = 1.0 

    def _create_voice_params(self):
        return texttospeech.VoiceSelectionParams(   
            language_code="pt-BR",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            name="pt-BR-Wavenet-B"
        )
    
    def _create_audio_config(self):
        return texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=self.speech_speed  
        )
    
    def synthesize_text(self, text):
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self._create_voice_params(),
            audio_config=self._create_audio_config()
        )
        
        self._save_and_play_audio(response.audio_content)
    
    def _save_and_play_audio(self, audio_content):
        with open(self.output_file, "wb") as out:
            out.write(audio_content)
        playsound(self.output_file)
        os.remove(self.output_file)

if __name__ == "__main__":
    notifier = TextToSpeechNotifier()
    notifier.speech_speed = 1.5 
    text = "Objeto detectado: Pessoa próxima à câmera."
    notifier.synthesize_text(text)
