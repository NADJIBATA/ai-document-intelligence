import { useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/context";
import { AuthLayout } from "../components/layout/AuthLayout";
import { Alert } from "../components/ui/Alert";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { errorMessage } from "../lib/errors";

// Same policy as the backend (app/schemas/auth.py).
const PASSWORD_MIN_LENGTH = 8;
const PASSWORD_MAX_LENGTH = 128;

function passwordProblem(password: string): string | null {
  if (password.length < PASSWORD_MIN_LENGTH) {
    return `At least ${PASSWORD_MIN_LENGTH} characters.`;
  }
  if (password.length > PASSWORD_MAX_LENGTH) {
    return `At most ${PASSWORD_MAX_LENGTH} characters.`;
  }
  if (!/\p{L}/u.test(password) || !/\d/.test(password)) {
    return "At least one letter and one digit.";
  }
  return null;
}

export function RegisterPage() {
  const { status, register } = useAuth();
  const navigate = useNavigate();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (status === "authenticated") {
    return <Navigate to="/" replace />;
  }

  const passwordError = password ? passwordProblem(password) : null;
  const confirmationError =
    confirmation && confirmation !== password ? "The passwords do not match." : null;
  const canSubmit =
    email.trim() !== "" && password !== "" && passwordError === null && confirmation === password;

  async function onSubmit(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await register(email.trim(), password, fullName);
      navigate("/", { replace: true });
    } catch (reason) {
      setError(errorMessage(reason));
      setSubmitting(false);
    }
  }

  return (
    <AuthLayout
      title="Create your account"
      subtitle="Your documents stay private: only you can search them."
      footer={
        <>
          Already registered?{" "}
          <Link to="/login" className="font-medium text-brand-700 hover:underline">
            Sign in
          </Link>
        </>
      }
    >
      <form onSubmit={(event) => void onSubmit(event)} className="space-y-4" noValidate>
        {error ? <Alert>{error}</Alert> : null}
        <Input
          label="Full name (optional)"
          name="full_name"
          autoComplete="name"
          maxLength={255}
          value={fullName}
          onChange={(event) => setFullName(event.target.value)}
        />
        <Input
          label="Email"
          type="email"
          name="email"
          autoComplete="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <Input
          label="Password"
          type="password"
          name="password"
          autoComplete="new-password"
          required
          hint={`At least ${PASSWORD_MIN_LENGTH} characters, with a letter and a digit.`}
          error={passwordError}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        <Input
          label="Confirm password"
          type="password"
          name="password_confirmation"
          autoComplete="new-password"
          required
          error={confirmationError}
          value={confirmation}
          onChange={(event) => setConfirmation(event.target.value)}
        />
        <Button type="submit" className="w-full" loading={submitting} disabled={!canSubmit}>
          Create account
        </Button>
      </form>
    </AuthLayout>
  );
}
