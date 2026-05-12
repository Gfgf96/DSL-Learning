"use client";

import { useTheme } from "next-themes";
import { Sun, Moon } from "lucide-react";

export function ThemeToggle() {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const activeTheme = resolvedTheme ?? theme;

  if (!activeTheme) {
    return (
      <button className="p-2.5 rounded-full bg-neutral-100 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 opacity-50 cursor-default">
        <div className="w-5 h-5" />
      </button>
    );
  }

  return (
    <button
      onClick={() => setTheme(activeTheme === "dark" ? "light" : "dark")}
      className="p-2.5 rounded-full bg-neutral-100 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-200 dark:hover:bg-neutral-800 transition-colors"
      aria-label="Toggle theme"
    >
      {activeTheme === "dark" ? (
        <Sun className="w-5 h-5 text-neutral-400 hover:text-white" />
      ) : (
        <Moon className="w-5 h-5 text-neutral-600 hover:text-black" />
      )}
    </button>
  );
}
