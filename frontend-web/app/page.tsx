import { Header } from "../components/Header";
import { Sidebar } from "../components/Sidebar";
import { Chat } from "../components/Chat";


export default function Page() {
	return (
		<div className="flex h-screen flex-col bg-black">
			<Header />

			<div className="flex flex-1 overflow-hidden">
				<Sidebar />
				<Chat />
			</div>
		</div>
	);
}
