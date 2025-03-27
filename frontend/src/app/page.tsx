"use client";
import { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Mic, Send, Square, Volume2, VolumeX } from "lucide-react";
import { motion } from "framer-motion";
import { AudioRecorder } from "@/utils/audioRecorder";
import { SpeechToText } from "@/utils/speechToText";
import { TextToSpeech } from "@/utils/textToSpeech";
import { GroqClient } from "@/utils/groqClient";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { systemPrompt } from "../../constants";

interface Message {
  role: "user" | "assistant";
  content: string;
}

type Language = 'en-US' | 'hi-IN';

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role:"assistant",
      content:systemPrompt
    }
  ]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState<Language>('en-US');
  const [isLoading, setIsLoading] = useState(false);
  const audioRecorderRef = useRef<AudioRecorder | null>(null);
  const speechToTextRef = useRef<SpeechToText | null>(null);
  const textToSpeechRef = useRef<TextToSpeech | null>(null);
  const groqClientRef = useRef<GroqClient | null>(null);

  useEffect(() => {
    textToSpeechRef.current = new TextToSpeech();
    return () => {
      if (textToSpeechRef.current) {
        textToSpeechRef.current.stop();
      }
    };
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const newMessage: Message = { role: "user" as const, content: input };
    setMessages(prev => [...prev, newMessage]);
    setInput("");
    setIsLoading(true);

    try {
      if (!groqClientRef.current) {
        groqClientRef.current = new GroqClient();
      }

      const assistantMessage: Message = { role: "assistant" as const, content: "" };
      setMessages(prev => [...prev, assistantMessage]);

      let fullResponse = '';
      await groqClientRef.current.chatCompletion(
        [...messages, newMessage],
        (chunk) => {
          fullResponse += chunk;
          setMessages(prev => {
            const newMessages = [...prev];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage.role === "assistant") {
              lastMessage.content = fullResponse;
            }
            return newMessages;
          });
        }
      );

      // Speak the response
      if (textToSpeechRef.current) {
        textToSpeechRef.current.setLanguage(selectedLanguage);
        textToSpeechRef.current.speak(fullResponse);
        setIsSpeaking(true);
      }
    } catch (error) {
      console.error('Failed to get response:', error);
      setMessages(prev => {
        const newMessages = [...prev];
        const lastMessage = newMessages[newMessages.length - 1];
        if (lastMessage.role === "assistant") {
          lastMessage.content = "Sorry, I encountered an error. Please try again.";
        }
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRecording = async () => {
    if (!isRecording) {
      try {
        audioRecorderRef.current = new AudioRecorder();
        await audioRecorderRef.current.startRecording();
        setIsRecording(true);
      } catch (error) {
        console.error('Failed to start recording:', error);
      }
    } else {
      try {
        if (audioRecorderRef.current) {
          const audioBlob = await audioRecorderRef.current.stopRecording();
          await uploadAudio(audioBlob);
          setIsRecording(false);
        }
      } catch (error) {
        console.error('Failed to stop recording:', error);
      }
    }
  };

  const toggleSpeechToText = () => {
    if (!isListening) {
      try {
        speechToTextRef.current = new SpeechToText();
        speechToTextRef.current.setLanguage(selectedLanguage);
        speechToTextRef.current.startListening(
          (text) => {
            setInput(text);
          },
          () => {
            // This callback is called after 3 seconds of silence
            if (input.trim()) {
              sendMessage();
            }
          }
        );
        setIsListening(true);
      } catch (error) {
        console.error('Failed to start speech recognition:', error);
      }
    } else {
      if (speechToTextRef.current) {
        speechToTextRef.current.stopListening();
        setIsListening(false);
      }
    }
  };

  const toggleSpeaking = () => {
    if (textToSpeechRef.current) {
      if (isSpeaking) {
        textToSpeechRef.current.stop();
      } else {
        const lastAssistantMessage = messages
          .filter(msg => msg.role === "assistant")
          .pop();
        if (lastAssistantMessage) {
          textToSpeechRef.current.setLanguage(selectedLanguage);
          textToSpeechRef.current.speak(lastAssistantMessage.content);
        }
      }
      setIsSpeaking(!isSpeaking);
    }
  };

  const handleLanguageChange = (lang: Language) => {
    setSelectedLanguage(lang);
    if (speechToTextRef.current) {
      speechToTextRef.current.setLanguage(lang);
    }
    if (textToSpeechRef.current) {
      textToSpeechRef.current.setLanguage(lang);
    }
  };

  const uploadAudio = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      if (data.success) {
        const newMessage: Message = {
          role: "user" as const,
          content: `[Audio Message](${data.path})`
        };
        setMessages([...messages, newMessage]);
      }
    } catch (error) {
      console.error('Failed to upload audio:', error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <Card className="w-full max-w-2xl p-4 space-y-4 bg-gray-800">
        <CardContent className="space-y-2 h-96 overflow-y-auto">
          {messages.slice(1).map((msg, index) => (
           
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-2 rounded-lg ${
                msg.role === "user" ? "bg-blue-600 ml-auto" : "bg-gray-700"
              }`}
            >
              {msg.content.startsWith('[Audio Message]') ? (
                <audio controls src={msg.content.match(/\((.*?)\)/)?.[1]} className="max-w-full" />
              ) : (
                msg.content
              )}
            </motion.div>
          ))}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="p-2 rounded-lg bg-gray-700"
            >
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-200" />
              </div>
            </motion.div>
          )}
        </CardContent>
        <div className="flex items-center space-x-2">
          <Button variant="ghost" onClick={toggleRecording}>
            {isRecording ? (
              <Square className="w-6 h-6 text-red-500" />
            ) : (
              <Mic className="w-6 h-6 text-white" />
            )}
          </Button>
          <Input
            className="flex-1 bg-gray-700 border-none"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage()}
            disabled={isLoading}
          />
          <Select value={selectedLanguage} onValueChange={handleLanguageChange}>
            <SelectTrigger className="w-[100px] bg-gray-700 border-none">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="en-US">English</SelectItem>
              <SelectItem value="hi-IN">हिंदी</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="ghost" onClick={toggleSpeechToText} disabled={isLoading}>
            <Mic className={`w-6 h-6 ${isListening ? "text-green-500" : "text-white"}`} />
          </Button>
          <Button variant="ghost" onClick={toggleSpeaking} disabled={isLoading}>
            {isSpeaking ? (
              <VolumeX className="w-6 h-6 text-red-500" />
            ) : (
              <Volume2 className="w-6 h-6 text-white" />
            )}
          </Button>
          <Button onClick={sendMessage} disabled={isLoading}>
            <Send className="w-6 h-6" />
          </Button>
        </div>
      </Card>
    </div>
  );
}
