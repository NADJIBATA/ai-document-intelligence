import { NavLink } from "react-router-dom";

import type { ConversationSummary } from "../../api/types";
import { classNames, formatDateTime } from "../../lib/format";
import { TrashIcon } from "../ui/Icons";

interface ConversationListProps {
  conversations: ConversationSummary[];
  onDelete: (conversation: ConversationSummary) => void;
  onNavigate?: () => void;
}

export function ConversationList({ conversations, onDelete, onNavigate }: ConversationListProps) {
  if (conversations.length === 0) {
    return <p className="px-3 py-6 text-center text-sm text-slate-500">No conversation yet.</p>;
  }
  return (
    <ul className="space-y-1">
      {conversations.map((conversation) => (
        <li key={conversation.id} className="group relative">
          <NavLink
            to={`/chat/${conversation.id}`}
            onClick={onNavigate}
            className={({ isActive }) =>
              classNames(
                "block rounded-lg px-3 py-2 pr-9 text-sm transition-colors",
                isActive ? "bg-brand-50 text-brand-900" : "text-slate-700 hover:bg-slate-100",
              )
            }
          >
            <span className="block truncate font-medium">{conversation.title}</span>
            <span className="block truncate text-xs text-slate-500">
              {formatDateTime(conversation.updated_at)}
            </span>
          </NavLink>
          <button
            type="button"
            onClick={() => onDelete(conversation)}
            className="absolute right-1.5 top-2 rounded-md p-1 text-slate-400 opacity-0 hover:bg-white hover:text-red-600 focus:opacity-100 group-hover:opacity-100"
            aria-label={`Delete conversation: ${conversation.title}`}
          >
            <TrashIcon width={16} height={16} />
          </button>
        </li>
      ))}
    </ul>
  );
}
