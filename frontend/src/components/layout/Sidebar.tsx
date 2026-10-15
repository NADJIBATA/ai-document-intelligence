import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import { useAuth } from "../../auth/context";
import { classNames } from "../../lib/format";
import {
  ChatIcon,
  CompareIcon,
  DashboardIcon,
  DocumentIcon,
  LogoutIcon,
  SparklesIcon,
} from "../ui/Icons";

interface NavItem {
  to: string;
  label: string;
  icon: ReactNode;
  end?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Dashboard", icon: <DashboardIcon />, end: true },
  { to: "/documents", label: "Documents", icon: <DocumentIcon /> },
  { to: "/chat", label: "Chat", icon: <ChatIcon /> },
  { to: "/intelligence", label: "Document Intelligence", icon: <SparklesIcon /> },
  { to: "/compare", label: "Compare Documents", icon: <CompareIcon /> },
];

interface SidebarProps {
  /** Called after a navigation (closes the drawer on small screens). */
  onNavigate?: () => void;
}

export function Sidebar({ onNavigate }: SidebarProps) {
  const { user, logout } = useAuth();

  return (
    <div className="flex h-full w-64 flex-col bg-slate-900 text-slate-300">
      <div className="flex items-center gap-3 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white">
          <DocumentIcon />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold text-white">Document Intelligence</p>
          <p className="text-xs text-slate-400">AI platform</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2" aria-label="Main">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            className={({ isActive }) =>
              classNames(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive ? "bg-slate-800 text-white" : "hover:bg-slate-800/60 hover:text-white",
              )
            }
          >
            {item.icon}
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-slate-800 px-4 py-4">
        <p className="truncate text-sm font-medium text-white" title={user?.email}>
          {user?.full_name || user?.email}
        </p>
        {user?.full_name ? <p className="truncate text-xs text-slate-400">{user.email}</p> : null}
        <button
          type="button"
          onClick={() => void logout()}
          className="mt-3 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"
        >
          <LogoutIcon width={18} height={18} />
          Sign out
        </button>
      </div>
    </div>
  );
}
