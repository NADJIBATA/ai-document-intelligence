import { Navigate, Outlet, useLocation } from "react-router-dom";

import { Spinner } from "../components/ui/Spinner";
import { useAuth } from "./context";

/** Renders the nested routes for signed-in users, redirects the others to /login. */
export function ProtectedRoute() {
  const { status } = useAuth();
  const location = useLocation();

  if (status === "loading") {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner label="Loading your workspace" />
      </div>
    );
  }
  if (status === "anonymous") {
    return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  }
  return <Outlet />;
}
