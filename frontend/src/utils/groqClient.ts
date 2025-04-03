import Groq from 'groq-sdk';

interface Message {
  role: "user" | "assistant";
  content: string;
}

export class GroqClient {
  private groq: Groq;
  private model: string = "qwen-2.5-coder-32b";

  constructor() {
    const apiKey = process.env.NEXT_PUBLIC_GROQ_API_KEY;
    if (!apiKey) {
      throw new Error('NEXT_PUBLIC_GROQ_API_KEY environment variable is not set');
    }
    this.groq = new Groq({
      apiKey,
      dangerouslyAllowBrowser: true
    });
  }

  
  async chatCompletion(messages: Message[], onStream?: (chunk: string) => void): Promise<string> {
    try {
      const chatCompletion = await this.groq.chat.completions.create({
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        model: this.model,
        temperature: 1,
        max_completion_tokens: 1024,
        top_p: 1,
        stream: true,
        stop: null
      });

      let fullResponse = '';
      let lastChunk = '';
      
      if (onStream) {
        for await (const chunk of chatCompletion) {
          const content = chunk.choices[0]?.delta?.content || '';
          if (content !== lastChunk) {
            fullResponse += content;
            onStream(content);
            lastChunk = content;
          }
        }
      } else {
        for await (const chunk of chatCompletion) {
          const content = chunk.choices[0]?.delta?.content || '';
          if (content !== lastChunk) {
            fullResponse += content;
            lastChunk = content;
          }
        }
      }

      return fullResponse;
    } catch (error) {
      console.error('Groq API Error:', error);
      throw error;
    }
  }

  setModel(model: string): void {
    this.model = model;
  }
} 