export function ErrorBanner({ message }: { message: string }) {
  return (
    <div role="alert" className="border-l-2 border-verdict bg-paper-raised px-4 py-3 text-sm text-ink">
      {message}
    </div>
  );
}
