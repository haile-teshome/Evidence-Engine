"use client";

import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { motion } from "motion/react";

import { cn } from "./utils";

// Tracks the active value + a per-Tabs layoutId so TabsTrigger can render a
// sliding active-indicator that animates from the old tab to the new one.
const TabsCtx = React.createContext<{ value?: string; layoutId: string }>({ layoutId: "" });

function Tabs({
  className,
  value,
  defaultValue,
  onValueChange,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Root>) {
  // Mirror the active value (works for both controlled and uncontrolled usage)
  // so triggers know which one is active without Radix exposing it.
  const [internal, setInternal] = React.useState<string | undefined>(
    (value as string) ?? (defaultValue as string),
  );
  const active = value !== undefined ? (value as string) : internal;
  const layoutId = React.useId();
  const handleChange = (v: string) => { setInternal(v); onValueChange?.(v); };
  return (
    <TabsCtx.Provider value={{ value: active, layoutId }}>
      <TabsPrimitive.Root
        data-slot="tabs"
        className={cn("flex flex-col gap-2", className)}
        value={value}
        defaultValue={defaultValue}
        onValueChange={handleChange}
        {...props}
      />
    </TabsCtx.Provider>
  );
}

function TabsList({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      className={cn(
        "bg-muted text-muted-foreground inline-flex h-9 w-fit items-center justify-center rounded-xl p-[3px] flex",
        className,
      )}
      {...props}
    />
  );
}

function TabsTrigger({
  className,
  children,
  value,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  const ctx = React.useContext(TabsCtx);
  const isActive = ctx.value !== undefined && ctx.value === value;
  return (
    <TabsPrimitive.Trigger
      data-slot="tabs-trigger"
      value={value}
      className={cn(
        "relative dark:data-[state=active]:text-foreground focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:outline-ring text-foreground dark:text-muted-foreground inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-xl border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-[color] focus-visible:ring-[3px] focus-visible:outline-1 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        className,
      )}
      {...props}
    >
      {/* Sliding active indicator — shared layout, animates between tabs.
          Uses the default light surface so text stays legible on every page. */}
      {isActive && (
        <motion.span
          layoutId={`${ctx.layoutId}-tab-indicator`}
          transition={{ type: "spring", stiffness: 440, damping: 36, mass: 0.7 }}
          className="absolute inset-0 rounded-xl bg-card shadow-sm dark:bg-input/30 border border-transparent dark:border-input"
        />
      )}
      <span className="relative z-10 inline-flex items-center gap-1.5">{children}</span>
    </TabsPrimitive.Trigger>
  );
}

function TabsContent({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return (
    <TabsPrimitive.Content
      data-slot="tabs-content"
      className={cn("flex-1 outline-none", className)}
      {...props}
    />
  );
}

export { Tabs, TabsList, TabsTrigger, TabsContent };
