import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";
import { cx } from "@/shared/ui/cx";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "icon";

const buttonVariants: Record<ButtonVariant, string> = {
  primary:
    "border-slate-900 bg-slate-900 text-white shadow-sm hover:bg-slate-800 hover:border-slate-800",
  secondary:
    "border-slate-300 bg-white text-slate-800 shadow-sm hover:bg-slate-50 hover:border-slate-400",
  ghost:
    "border-transparent bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-950",
  danger:
    "border-red-200 bg-red-50 text-red-700 hover:bg-red-100 hover:border-red-300",
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-10 px-4 text-sm",
  icon: "h-9 w-9 p-0",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export function Button({
  variant = "secondary",
  size = "md",
  className,
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-md border font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        buttonVariants[variant],
        buttonSizes[size],
        className
      )}
      {...props}
    />
  );
}

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: "default" | "good" | "warning" | "info";
}

export function Badge({ tone = "default", className, ...props }: BadgeProps) {
  const toneClass = {
    default: "border-slate-200 bg-slate-100 text-slate-700",
    good: "border-emerald-200 bg-emerald-50 text-emerald-700",
    warning: "border-amber-200 bg-amber-50 text-amber-800",
    info: "border-cyan-200 bg-cyan-50 text-cyan-800",
  }[tone];
  return (
    <span
      className={cx(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium",
        toneClass,
        className
      )}
      {...props}
    />
  );
}

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cx("rounded-lg border border-slate-200 bg-white shadow-sm", className)}
      {...props}
    />
  );
}

export function PageState({
  title,
  detail,
  tone = "default",
}: {
  title: string;
  detail?: ReactNode;
  tone?: "default" | "error";
}) {
  return (
    <Panel
      className={cx(
        "p-6 text-center",
        tone === "error" ? "border-red-200 bg-red-50 text-red-800" : "text-slate-600"
      )}
    >
      <p className="m-0 text-sm font-semibold">{title}</p>
      {detail ? <div className="mt-1 text-sm">{detail}</div> : null}
    </Panel>
  );
}

export function TabButton({
  active,
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      type="button"
      className={cx(
        "h-9 rounded-md px-3 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500",
        active
          ? "bg-white text-slate-950 shadow-sm"
          : "text-slate-500 hover:bg-white/70 hover:text-slate-800",
        className
      )}
      {...props}
    />
  );
}
