export class TextToSpeech {
  private synthesis: SpeechSynthesis;
  private utterance: SpeechSynthesisUtterance | null = null;
  private currentLang: string = 'en-US';
  private voices: SpeechSynthesisVoice[] = [];

  constructor() {
    this.synthesis = window.speechSynthesis;
    this.loadVoices();
  }

  private loadVoices(): void {
    // Load voices when they become available
    const loadVoicesList = () => {
      this.voices = this.synthesis.getVoices();
    };

    if (this.synthesis.getVoices().length > 0) {
      loadVoicesList();
    } else {
      this.synthesis.onvoiceschanged = loadVoicesList;
    }
  }

  setLanguage(lang: 'en-US' | 'hi-IN'): void {
    this.currentLang = lang;
  }

  private getVoiceForLanguage(lang: string): SpeechSynthesisVoice | undefined {
    // First try to find an exact match for the language
    let voice = this.voices.find(v => v.lang === lang);
    
    // If no exact match, try to find a voice that starts with the language code
    if (!voice) {
      voice = this.voices.find(v => v.lang.startsWith(lang.split('-')[0]));
    }

    // If still no match, use the first available voice
    if (!voice && this.voices.length > 0) {
      voice = this.voices[0];
    }

    return voice;
  }

  speak(text: string, onEnd?: () => void): void {
    // Cancel any ongoing speech
    if (this.utterance) {
      this.synthesis.cancel();
    }

    this.utterance = new SpeechSynthesisUtterance(text);
    this.utterance.lang = this.currentLang;
    
    // Get appropriate voice for the current language
    const voice = this.getVoiceForLanguage(this.currentLang);
    if (voice) {
      this.utterance.voice = voice;
    }

    // Adjust speech parameters based on language
    if (this.currentLang === 'hi-IN') {
      this.utterance.rate = 0.9; // Slightly slower for Hindi
      this.utterance.pitch = 1;
    } else {
      this.utterance.rate = 1;
      this.utterance.pitch = 1;
    }

    this.utterance.volume = 1;

    if (onEnd) {
      this.utterance.onend = onEnd;
    }

    this.synthesis.speak(this.utterance);
  }

  stop(): void {
    if (this.utterance) {
      this.synthesis.cancel();
      this.utterance = null;
    }
  }

  isSpeaking(): boolean {
    return this.synthesis.speaking;
  }

  getAvailableVoices(): SpeechSynthesisVoice[] {
    return this.voices;
  }
} 