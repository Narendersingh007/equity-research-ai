export function Sidebar() {
	return (
		<aside className="w-72 border-r border-zinc-800 p-4 text-xs text-zinc-400 space-y-3">
			<div>
				<div className="font-semibold mb-1">Neural Engine</div>
				<div>
					Status: <span className="text-green-400">Active</span>
				</div>
				<div>Index: finance-news</div>
				<div>Retrieval: Top-10 chunks</div>
			</div>

			<div className="pt-4">
				<div className="font-semibold mb-1">Coverage Universe</div>
				<div className="grid grid-cols-3 gap-1 text-zinc-500">
					{[
						"AAPL",
						"MSFT",
						"GOOGL",
						"AMZN",
						"NVDA",
						"META",
						"TSLA",
						"BRK.B",
					].map((t) => (
						<span key={t}>{t}</span>
					))}
				</div>
			</div>
		</aside>
	);
}
