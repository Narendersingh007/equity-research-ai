"use client";

import { useEffect, useRef, useState } from "react";

type Message = {
	role: "user" | "assistant";
	content: string;
	timestamp: string;
	sources?: {
		sec?: string[];
		news?: string[];
	};
};

const SUGGESTED_QUESTIONS = [
	"What are the key risks disclosed in Apple's latest 10-K?",
	"Summarize NVIDIA's revenue growth and AI segment performance",
	"Compare Microsoft and Google's cloud business metrics",
	"What is Tesla's current cash position and debt profile?",
];

export function Chat() {
	const [messages, setMessages] = useState<Message[]>([]);
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false);
	const [showShadow, setShowShadow] = useState(false);
	const bottomRef = useRef<HTMLDivElement>(null);

	/* ================= AUTO SCROLL ================= */
	useEffect(() => {
		bottomRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages, loading]);

	/* ================= INPUT SHADOW ================= */
	useEffect(() => {
		const handler = () => setShowShadow(true);
		window.addEventListener("scroll", handler);
		return () => window.removeEventListener("scroll", handler);
	}, []);

	async function sendMessage(text?: string) {
		const query = text ?? input;
		if (!query.trim() || loading) return;

		const now = new Date().toLocaleTimeString([], {
			hour: "2-digit",
			minute: "2-digit",
		});

		const userMessage: Message = {
			role: "user",
			content: query,
			timestamp: now,
		};

		setMessages((prev) => [...prev, userMessage]);
		setInput("");
		setLoading(true);

		try {
			const res = await fetch("http://localhost:8000/chat", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ query }),
			});

			const data = await res.json();

			setMessages((prev) => [
				...prev,
				{
					role: "assistant",
					content: data.answer,
					sources: data.sources,
					timestamp: now,
				},
			]);
		} catch {
			setMessages((prev) => [
				...prev,
				{
					role: "assistant",
					content: "⚠️ Failed to contact backend.",
					timestamp: now,
				},
			]);
		} finally {
			setLoading(false);
		}
	}

	return (
		<main className="flex flex-1 flex-col bg-black">
			{/* ================= MESSAGE AREA ================= */}
			<div className="flex-1 overflow-y-auto px-6 py-12 flex justify-center">
				<div className="w-full max-w-[760px] space-y-14">
					{/* EMPTY STATE */}
					{messages.length === 0 && (
						<div className="text-center text-zinc-500 text-sm space-y-4 pt-24">
							<div className="text-zinc-400 font-medium">
								Equity Research AI Agent
							</div>
							<p>
								Ask questions about SEC filings, financials, and market news for
								top global companies.
							</p>

							{/* SUGGESTED QUESTIONS */}
							<div className="flex flex-wrap justify-center gap-2 pt-6">
								{SUGGESTED_QUESTIONS.map((q) => (
									<button
										key={q}
										onClick={() => sendMessage(q)}
										className="text-xs px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-300 hover:border-green-600/40 hover:text-green-400 transition"
									>
										{q}
									</button>
								))}
							</div>
						</div>
					)}

					{/* MESSAGES */}
					{messages.map((msg, idx) => (
						<div
							key={idx}
							className={`flex ${
								msg.role === "user" ? "justify-end" : "justify-start"
							}`}
						>
							{/* USER */}
							{msg.role === "user" && (
								<div className="max-w-[520px] space-y-1 text-right">
									<div className="rounded-2xl px-5 py-3 text-sm bg-green-600/20 text-green-300 border border-green-600/40 shadow-sm">
										{msg.content}
									</div>
									<div className="text-[10px] text-zinc-500">
										{msg.timestamp}
									</div>
								</div>
							)}

							{/* ASSISTANT */}
							{msg.role === "assistant" && (
								<div className="max-w-[620px] bg-zinc-900/90 border border-zinc-800 rounded-2xl shadow-md flex flex-col max-h-[540px]">
									{/* STICKY HEADER */}
									<div className="sticky top-0 z-10 flex justify-between items-center px-6 py-4 bg-zinc-900/95 backdrop-blur border-b border-zinc-800 rounded-t-2xl text-xs text-zinc-400">
										<div className="flex items-center gap-2">
											<span className="h-2 w-2 rounded-full bg-green-500" />
											Research Brief
										</div>
										<span>{msg.timestamp}</span>
									</div>

									{/* CONTENT */}
									<div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
										<div className="text-sm text-zinc-200 leading-7 whitespace-pre-line">
											{msg.content}
										</div>

										{/* COLLAPSIBLE SOURCES */}
										{msg.sources && (
											<details className="border-t border-zinc-800 pt-4 text-xs text-zinc-400">
												<summary className="cursor-pointer font-semibold text-zinc-300 mb-2">
													Sources
												</summary>

												<div className="space-y-3 pl-4">
													{msg.sources.sec?.length && (
														<div>
															<div className="font-medium text-zinc-300 mb-1">
																SEC Filings
															</div>
															<ul className="list-disc list-inside space-y-0.5">
																{msg.sources.sec.map((s, i) => (
																	<li key={i}>{s}</li>
																))}
															</ul>
														</div>
													)}

													{msg.sources.news?.length && (
														<div>
															<div className="font-medium text-zinc-300 mb-1">
																News
															</div>
															<ul className="list-disc list-inside space-y-0.5">
																{msg.sources.news.map((s, i) => (
																	<li key={i}>{s}</li>
																))}
															</ul>
														</div>
													)}
												</div>
											</details>
										)}
									</div>
								</div>
							)}
						</div>
					))}

					{loading && (
						<div className="flex items-center gap-2 text-zinc-500 text-sm">
							<span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
							Analyzing SEC filings and live market news…
						</div>
					)}

					<div ref={bottomRef} />
				</div>
			</div>

			{/* ================= INPUT BAR ================= */}
			<div
				className={`border-t border-zinc-800 bg-black/80 backdrop-blur px-6 py-4 ${
					showShadow ? "shadow-[0_-8px_30px_rgba(0,0,0,0.6)]" : ""
				}`}
			>
				<div className="flex gap-3 max-w-[760px] mx-auto">
					<input
						value={input}
						onChange={(e) => setInput(e.target.value)}
						onKeyDown={(e) => e.key === "Enter" && sendMessage()}
						placeholder="Ask about SEC filings, financials, or market news…"
						className="flex-1 rounded-md bg-zinc-900 px-4 py-2.5 text-sm text-zinc-200 outline-none border border-zinc-800 focus:border-green-600/50"
						disabled={loading}
					/>
					<button
						onClick={() => sendMessage()}
						disabled={loading}
						className="rounded-md bg-green-600 px-5 text-sm font-medium text-black disabled:opacity-50"
					>
						Send
					</button>
				</div>

				<div className="text-xs text-zinc-500 mt-2 text-center">
					Press Enter to send · Shift + Enter for new line
				</div>
			</div>
		</main>
	);
}
