"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Send,
  Paperclip,
  FileText,
  Image as ImageIcon,
  Loader2,
  Upload,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Trash2,
} from "lucide-react";
import {
  askQuestion,
  uploadDocument,
  fileToBase64,
  getRagHealth,
  type Source,
} from "@/lib/rag-api";
import { getChatContext } from "@/lib/api";
import type { SimulationResponse } from "@/lib/types";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  imageB64?: string;       // user-attached image shown in the bubble
  sources?: Source[];      // assistant citations
  error?: boolean;
}

interface IndexedDoc {
  name: string;
  chunks: number;
}

interface ChatAdvisorProps {
  open: boolean;
  onClose: () => void;
  recommendationData?: SimulationResponse | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function uid() {
  return Math.random().toString(36).slice(2);
}

function isImage(file: File) {
  return file.type.startsWith("image/");
}

function isPdf(file: File) {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

const ACCEPTED = ".pdf,.png,.jpg,.jpeg,.webp";

// ── Sub-components ────────────────────────────────────────────────────────────

function SourceAccordion({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  return (
    <div className="mt-2 rounded-md border border-neutral-700/60 overflow-hidden text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-1.5 bg-neutral-800/60 text-neutral-400 hover:text-neutral-200 transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <BookOpen size={11} />
          {sources.length} source{sources.length > 1 ? "s" : ""}
        </span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="divide-y divide-neutral-700/40">
              {sources.map((s, i) => (
                <div key={i} className="px-3 py-2 bg-neutral-900/40">
                  <p className="font-medium text-cyan-400/80 mb-0.5 truncate">{s.source}</p>
                  <p className="text-neutral-400 leading-relaxed">{s.excerpt}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-2.5 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && (
        <div className="shrink-0 mt-0.5 w-7 h-7 rounded-full bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center text-white text-[10px] font-bold">
          AI
        </div>
      )}

      <div className={`max-w-[82%] flex flex-col ${isUser ? "items-end" : "items-start"}`}>
        {/* Attached image preview */}
        {msg.imageB64 && (
          <div className="mb-1.5 rounded-lg overflow-hidden border border-neutral-700/60 max-w-[180px]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/jpeg;base64,${msg.imageB64}`}
              alt="Attached screenshot"
              className="w-full object-contain"
            />
          </div>
        )}

        {/* Text bubble */}
        <div
          className={`px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-cyan-600/20 border border-cyan-500/25 text-cyan-50 rounded-tr-sm"
              : msg.error
              ? "bg-red-500/10 border border-red-500/25 text-red-300 rounded-tl-sm"
              : "bg-neutral-800/80 border border-neutral-700/50 text-neutral-100 rounded-tl-sm"
          }`}
        >
          {msg.text}
        </div>

        {/* Sources */}
        {msg.sources && <SourceAccordion sources={msg.sources} />}
      </div>

      {isUser && (
        <div className="shrink-0 mt-0.5 w-7 h-7 rounded-full bg-neutral-700 flex items-center justify-center text-neutral-300 text-[10px] font-bold">
          You
        </div>
      )}
    </motion.div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ChatAdvisor({ open, onClose, recommendationData }: ChatAdvisorProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: "assistant",
      text:
        "Hello! I'm your JoSAA counselling advisor.\n\n" +
        "You can:\n" +
        "• Ask me anything about the counselling process, quotas, or document requirements\n" +
        "• Upload a PDF rulebook or brochure and ask specific questions about it\n" +
        "• Attach a screenshot of your counselling portal status and I'll read it for you\n\n" +
        "How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [attachedImage, setAttachedImage] = useState<File | null>(null);
  const [attachedImageB64, setAttachedImageB64] = useState<string | null>(null);
  const [indexedDocs, setIndexedDocs] = useState<IndexedDoc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [healthMessage, setHealthMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`;
  }, [input]);

  useEffect(() => {
    if (!open) return;
    let active = true;
    getRagHealth()
      .then((health) => {
        if (!active) return;
        setHealthMessage(
          health.google_api_key_configured ? null : "RAG missing Google API key."
        );
      })
      .catch(() => {
        if (active) setHealthMessage("RAG unavailable.");
      });
    return () => {
      active = false;
    };
  }, [open]);

  // ── Upload a PDF for indexing ─────────────────────────────────────────────

  const handleDocUpload = useCallback(
    async (file: File) => {
      if (!isPdf(file) && !isImage(file)) {
        setUploadError("Only PDF and image files are supported.");
        return;
      }
      setUploading(true);
      setUploadError(null);
      try {
        const result = await uploadDocument(file);
        setIndexedDocs((prev) => [
          ...prev,
          { name: result.source, chunks: result.chunks_added },
        ]);
        setMessages((prev) => [
          ...prev,
          {
            id: uid(),
            role: "assistant",
            text: `✅ Indexed **${result.source}** — ${result.chunks_added} chunk${result.chunks_added !== 1 ? "s" : ""} added to context. You can now ask questions about it.`,
          },
        ]);
      } catch (e) {
        setUploadError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    []
  );

  // ── Attach an image to the next message ──────────────────────────────────

  const handleImageAttach = useCallback(async (file: File) => {
    if (!isImage(file)) return;
    const b64 = await fileToBase64(file);
    setAttachedImage(file);
    setAttachedImageB64(b64);
  }, []);

  // ── Send message ──────────────────────────────────────────────────────────

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || thinking) return;

    const userMsg: Message = {
      id: uid(),
      role: "user",
      text,
      imageB64: attachedImageB64 ?? undefined,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setAttachedImage(null);
    setAttachedImageB64(null);
    setThinking(true);

    try {
      let finalMlContext: string | undefined = undefined;
      if (recommendationData) {
        try {
          const { context_block } = await getChatContext(text, recommendationData);
          if (context_block) finalMlContext = context_block;
        } catch (e) {
          console.warn("Failed to get ml context", e);
        }
      }

      const res = await askQuestion(text, attachedImageB64 ?? undefined, finalMlContext);
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          text: res.answer,
          sources: res.sources,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          text:
            e instanceof Error
              ? `Error: ${e.message}`
              : "Something went wrong. Please try again.",
          error: true,
        },
      ]);
    } finally {
      setThinking(false);
    }
  }, [input, thinking, attachedImageB64, recommendationData]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Drag & drop for doc upload ─────────────────────────────────────────────

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (isImage(file)) {
        handleImageAttach(file);
      } else {
        handleDocUpload(file);
      }
    },
    [handleDocUpload, handleImageAttach]
  );

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            key="chat-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
          />

          {/* Drawer — slides from the left */}
          <motion.aside
            key="chat-drawer"
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 280 }}
            className="fixed left-0 top-0 h-full w-full sm:w-[480px] bg-neutral-950 border-r border-neutral-800 z-50 flex flex-col"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {/* ── Header ─────────────────────────────────────────────────────── */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-neutral-800 shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center shadow-lg shadow-cyan-900/30">
                  <BookOpen size={14} className="text-white" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white leading-tight">
                    Counselling Advisor
                  </p>
                  <p className="text-[11px] text-neutral-500 leading-tight">
                    Powered by Gemini + RAG
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-md hover:bg-neutral-900 text-neutral-500 hover:text-white transition-colors"
              >
                <X size={16} />
              </button>
            </div>

            {/* ── Indexed docs strip ──────────────────────────────────────────── */}
            {indexedDocs.length > 0 && (
              <div className="px-4 py-2.5 border-b border-neutral-800 flex items-center gap-2 flex-wrap shrink-0 bg-neutral-900/40">
                <span className="text-[11px] text-neutral-500 font-medium">Indexed:</span>
                {indexedDocs.map((doc, i) => (
                  <span
                    key={i}
                    className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400"
                  >
                    <FileText size={9} />
                    {doc.name}
                    <span className="text-cyan-600 ml-0.5">·{doc.chunks}</span>
                  </span>
                ))}
              </div>
            )}

            {/* ── Messages ────────────────────────────────────────────────────── */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 scroll-smooth">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}

              {/* Thinking indicator */}
              {thinking && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-2.5 justify-start"
                >
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center text-white text-[10px] font-bold shrink-0 mt-0.5">
                    AI
                  </div>
                  <div className="px-3.5 py-3 rounded-2xl rounded-tl-sm bg-neutral-800/80 border border-neutral-700/50 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-bounce [animation-delay:0ms]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-bounce [animation-delay:150ms]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-bounce [animation-delay:300ms]" />
                  </div>
                </motion.div>
              )}

              <div ref={bottomRef} />
            </div>

            {/* ── Attached image preview ──────────────────────────────────────── */}
            {attachedImage && (
              <div className="mx-4 mb-2 flex items-center gap-2 px-3 py-2 rounded-lg bg-neutral-800/70 border border-neutral-700/50">
                <ImageIcon size={13} className="text-cyan-400 shrink-0" />
                <span className="text-xs text-neutral-300 truncate flex-1">{attachedImage.name}</span>
                <button
                  onClick={() => { setAttachedImage(null); setAttachedImageB64(null); }}
                  className="text-neutral-500 hover:text-red-400 transition-colors"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            )}

            {/* ── Upload status / error ───────────────────────────────────────── */}
            {(uploading || uploadError || healthMessage) && (
              <div className="mx-4 mb-2">
                {uploading ? (
                  <div className="flex items-center gap-2 text-xs text-neutral-400 px-3 py-2 rounded-lg bg-neutral-800/60">
                    <Loader2 size={12} className="animate-spin text-cyan-400" />
                    Indexing document…
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-xs text-red-400 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20">
                    <AlertCircle size={12} />
                    {uploadError ?? healthMessage}
                  </div>
                )}
              </div>
            )}

            {/* ── Input area ──────────────────────────────────────────────────── */}
            <div className="px-4 pb-4 pt-2 border-t border-neutral-800 shrink-0">
              <div className="flex items-end gap-2">
                {/* Action buttons */}
                <div className="flex flex-col gap-1.5 pb-1">
                  {/* Upload PDF for indexing */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={ACCEPTED}
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      if (isImage(f)) { handleImageAttach(f); }
                      else { handleDocUpload(f); }
                      e.target.value = "";
                    }}
                  />
                  <button
                    title="Upload PDF or image to index"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                    className="p-2 rounded-lg text-neutral-500 hover:text-cyan-400 hover:bg-neutral-800 transition-colors disabled:opacity-40"
                  >
                    {uploading ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <Upload size={16} />
                    )}
                  </button>
                  {/* Attach image to question */}
                  <input
                    ref={imageInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleImageAttach(f);
                      e.target.value = "";
                    }}
                  />
                  <button
                    title="Attach screenshot to your message"
                    onClick={() => imageInputRef.current?.click()}
                    className={`p-2 rounded-lg transition-colors ${
                      attachedImage
                        ? "text-cyan-400 bg-cyan-500/10"
                        : "text-neutral-500 hover:text-cyan-400 hover:bg-neutral-800"
                    }`}
                  >
                    <Paperclip size={16} />
                  </button>
                </div>

                {/* Textarea */}
                <div className="flex-1 relative">
                  <textarea
                    ref={textareaRef}
                    id="chat-input"
                    rows={1}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about counselling, quotas, documents…"
                    className="w-full resize-none rounded-xl bg-neutral-900 border border-neutral-700 text-sm text-neutral-100 placeholder-neutral-600
                      px-4 py-3 pr-12 focus:outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/20 transition-colors
                      min-h-[44px] max-h-[120px] overflow-y-auto"
                  />
                  <button
                    onClick={handleSend}
                    disabled={!input.trim() || thinking}
                    className="absolute right-2.5 bottom-2 p-1.5 rounded-lg bg-cyan-600 text-white
                      hover:bg-cyan-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                  >
                    {thinking ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Send size={14} />
                    )}
                  </button>
                </div>
              </div>

              <p className="text-[10px] text-neutral-600 mt-2 text-center">
                Drag & drop a PDF or image anywhere · Press Enter to send · Shift+Enter for newline
              </p>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
