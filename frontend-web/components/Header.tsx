export function Header() {
	return (
		<div className="border-b border-zinc-800 px-6 py-3 flex items-center gap-4">
			<h1 className="text-sm font-semibold">Equity Research AI Agent</h1>

			<span className="text-xs text-zinc-500">Live RAG Analysis</span>
			<span className="text-xs text-zinc-500">SEC Filings (10-K)</span>
			<span className="text-xs text-zinc-500">News Intelligence</span>

			<div className="ml-auto flex items-center gap-2 text-xs text-green-400">
				<span className="h-2 w-2 rounded-full bg-green-500" />
				Online
			</div>
		</div>
	);
}
