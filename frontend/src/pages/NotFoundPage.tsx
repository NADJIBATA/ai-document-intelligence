import { Link } from "react-router-dom";

import { PageContainer } from "../components/layout/PageHeader";
import { EmptyState } from "../components/ui/EmptyState";
import { AlertIcon } from "../components/ui/Icons";

export function NotFoundPage() {
  return (
    <PageContainer>
      <EmptyState
        icon={<AlertIcon />}
        title="Page not found"
        description="The page you are looking for does not exist or has been moved."
        action={
          <Link
            to="/"
            className="inline-flex rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
          >
            Back to the dashboard
          </Link>
        }
      />
    </PageContainer>
  );
}
