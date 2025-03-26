export class SpeechToText {
  private recognition: SpeechRecognition | null = null;
  private currentLang: string = 'en-US';
  private silenceTimeout: NodeJS.Timeout | null = null;
  private onSilenceCallback: (() => void) | null = null;
  private SILENCE_THRESHOLD = 3000; // 3 seconds of silence

  constructor() {
    if ('webkitSpeechRecognition' in window) {
      const SpeechRecognition = (window as Window).webkitSpeechRecognition;
      this.recognition = new SpeechRecognition();
      
      if (this.recognition) {
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = this.currentLang;
      }
    }
  }

  setLanguage(lang: 'en-US' | 'hi-IN'): void {
    this.currentLang = lang;
    if (this.recognition) {
      this.recognition.lang = lang;
    }
  }

  startListening(onResult: (text: string) => void, onSilence?: () => void): void {
    if (!this.recognition) {
      throw new Error('Speech recognition not supported in this browser');
    }

    this.onSilenceCallback = onSilence || null;

    this.recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = Array.from(event.results)
        .map(result => result[0].transcript)
        .join('');

      onResult(transcript);

      // Reset silence timeout on new speech
      if (this.silenceTimeout) {
        clearTimeout(this.silenceTimeout);
      }

      // Set new silence timeout
      this.silenceTimeout = setTimeout(() => {
        if (this.onSilenceCallback) {
          this.onSilenceCallback();
        }
        this.stopListening();
      }, this.SILENCE_THRESHOLD);
    };

    this.recognition.onerror = (event: SpeechRecognitionError) => {
      console.error('Speech recognition error:', event.error);
    };

    this.recognition.start();
  }

  stopListening(): void {
    if (this.recognition) {
      this.recognition.stop();
    }
    if (this.silenceTimeout) {
      clearTimeout(this.silenceTimeout);
      this.silenceTimeout = null;
    }
    this.onSilenceCallback = null;
  }

  isListening(): boolean {
    return this.recognition?.state === 'listening' || false;
  }

  getCurrentLanguage(): string {
    return this.currentLang;
  }

  setSilenceThreshold(ms: number): void {
    this.SILENCE_THRESHOLD = ms;
  }
} 