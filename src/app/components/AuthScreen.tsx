import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from "motion/react";
import { useAuth } from "../lib/auth";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Loader2, ShieldCheck } from "lucide-react";

const REVIEW_TYPES = ["systematic reviews", "scoping reviews", "meta-analyses", "evidence syntheses"];

const EASE = [0.16, 1, 0.3, 1] as const;
const blurIn = (delay = 0) => ({
  initial: { opacity: 0, y: 22, filter: "blur(12px)" },
  animate: { opacity: 1, y: 0, filter: "blur(0px)" },
  transition: { duration: 0.8, ease: EASE, delay },
});

// Living background: a drifting network of nodes (studies) connected when close
// (an evidence/citation web), gently repelled by the pointer. Canvas + rAF, no
// dependency. Draws a single static frame when reduced motion is requested.
function ParticleField() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, raf = 0;
    let pts: { x: number; y: number; vx: number; vy: number }[] = [];
    const mouse = { x: -9999, y: -9999 };
    const LINK = 132;

    function resize() {
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = Math.floor(W * DPR); canvas.height = Math.floor(H * DPR);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      const count = Math.max(28, Math.min(96, Math.floor((W * H) / 16000)));
      pts = Array.from({ length: count }, () => ({
        x: Math.random() * W, y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
      }));
    }

    function frame() {
      ctx.clearRect(0, 0, W, H);
      for (const p of pts) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
        const dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
        if (d2 < 130 * 130) {
          const d = Math.sqrt(d2) || 1, f = (130 - d) / 130 * 0.9;
          p.x += (dx / d) * f; p.y += (dy / d) * f;
        }
      }
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const a = pts[i], b = pts[j];
          const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
          if (d < LINK) {
            ctx.strokeStyle = `rgba(13,148,136,${(1 - d / LINK) * 0.22})`;
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      }
      for (const p of pts) {
        ctx.fillStyle = "rgba(13,148,136,0.55)";
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.7, 0, Math.PI * 2); ctx.fill();
      }
      if (!reduce) raf = requestAnimationFrame(frame);
    }

    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
    };
    const onLeave = () => { mouse.x = -9999; mouse.y = -9999; };

    resize();
    frame();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerleave", onLeave);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", onLeave);
    };
  }, []);
  return <canvas ref={ref} className="absolute inset-0 h-full w-full" aria-hidden="true" />;
}

export function AuthScreen() {
  const { login, signup } = useAuth();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ti, setTi] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTi(v => (v + 1) % REVIEW_TYPES.length), 2200);
    return () => clearInterval(t);
  }, []);

  // Pointer tilt for the auth card only.
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 60, damping: 18, mass: 0.4 });
  const sy = useSpring(my, { stiffness: 60, damping: 18, mass: 0.4 });
  const onPointer = (e: React.PointerEvent) => {
    mx.set(e.clientX / window.innerWidth - 0.5);
    my.set(e.clientY / window.innerHeight - 0.5);
  };
  const cardRotX = useTransform(sy, v => v * -6);
  const cardRotY = useTransform(sx, v => v * 7);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "signup") await signup(email, password, name);
      else await login(email, password);
    } catch (err: any) {
      setError(err?.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div onPointerMove={onPointer} className="relative min-h-screen overflow-hidden bg-gradient-to-b from-background via-background to-muted/20" style={{ perspective: 1200 }}>
      {/* Living network background */}
      <ParticleField />

      <div className="relative z-10 min-h-screen grid lg:grid-cols-2">
        {/* Left: hero */}
        <div className="hidden lg:flex flex-col justify-center px-14 xl:px-24">
          <div className="relative max-w-xl">
            <div className="hero-glow -z-10" style={{ width: "34rem", height: "22rem", left: "-4rem", top: "-2rem" }} aria-hidden="true" />

            <h1 className="text-5xl xl:text-6xl font-semibold leading-[1.05] tracking-tight">
              <motion.span {...blurIn(0.05)} className="block">Your entire</motion.span>
              <motion.span {...blurIn(0.15)} className="block"><span className="glow-word">systematic review</span>,</motion.span>
              <motion.span {...blurIn(0.25)} className="block">in one place.</motion.span>
            </h1>

            <motion.p {...blurIn(0.4)} className="mt-6 text-lg text-muted-foreground max-w-md">
              Search, screen, extract, appraise, and write up, with AI that keeps every decision auditable.
            </motion.p>

            <motion.div {...blurIn(0.5)} className="mt-3 text-sm text-muted-foreground/90 flex items-center gap-1.5">
              <span>Built for</span>
              <span className="relative inline-flex items-center h-6 min-w-[9.5rem]">
                <AnimatePresence mode="wait">
                  <motion.span key={ti}
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.3 }} className="absolute inset-y-0 left-0 flex items-center font-medium text-primary">
                    {REVIEW_TYPES[ti]}
                  </motion.span>
                </AnimatePresence>
              </span>
            </motion.div>
          </div>
        </div>

        {/* Right: auth card with pointer tilt */}
        <div className="flex items-center justify-center px-4 py-10">
          <motion.div
            initial={{ opacity: 0, y: 26, scale: 0.97, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
            transition={{ duration: 0.8, ease: EASE, delay: 0.15 }}
            className="w-full max-w-sm" style={{ transformStyle: "preserve-3d" }}>
            <motion.div style={{ rotateX: cardRotX, rotateY: cardRotY, transformStyle: "preserve-3d" }}
              className="relative rounded-2xl border border-border/60 bg-card/70 backdrop-blur-xl shadow-2xl p-6 sm:p-7">
              <div className="card-sheen" aria-hidden="true" />
              <div className="mb-5">
                <AnimatePresence mode="wait">
                  <motion.h2 key={mode}
                    initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.25 }}
                    className="text-xl font-semibold tracking-tight">
                    {mode === "signup" ? "Create your account" : "Welcome back"}
                  </motion.h2>
                </AnimatePresence>
                <p className="text-sm text-muted-foreground mt-1">
                  {mode === "signup" ? "Set up a local account to save your reviews." : "Sign in to your local account."}
                </p>
              </div>

              <form onSubmit={submit} className="space-y-3">
                <AnimatePresence initial={false}>
                  {mode === "signup" && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.25 }} className="overflow-hidden">
                      <div className="space-y-1.5 pb-1">
                        <Label htmlFor="name">Name</Label>
                        <Input id="name" value={name} onChange={e => setName(e.target.value)} placeholder="Your name" autoComplete="name" />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="space-y-1.5">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" required value={email} onChange={e => setEmail(e.target.value)}
                    placeholder="you@example.com" autoComplete="email" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="password">Password</Label>
                  <Input id="password" type="password" required value={password} onChange={e => setPassword(e.target.value)}
                    placeholder={mode === "signup" ? "At least 8 characters" : "Your password"}
                    autoComplete={mode === "signup" ? "new-password" : "current-password"} />
                </div>

                {error && (
                  <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
                    className="text-sm text-rose-600 dark:text-rose-400">{error}</motion.p>
                )}

                <motion.div whileHover={{ scale: 1.015 }} whileTap={{ scale: 0.985 }}>
                  <Button type="submit" className="w-full mt-1" disabled={busy}>
                    {busy && <Loader2 className="size-4 mr-2 animate-spin" />}
                    {mode === "signup" ? "Create account" : "Log in"}
                  </Button>
                </motion.div>
              </form>

              <div className="text-center text-sm text-muted-foreground mt-5">
                {mode === "signup" ? (
                  <>Already have an account?{" "}
                    <button type="button" className="text-primary hover:underline font-medium" onClick={() => { setMode("login"); setError(""); }}>Log in</button>
                  </>
                ) : (
                  <>New here?{" "}
                    <button type="button" className="text-primary hover:underline font-medium" onClick={() => { setMode("signup"); setError(""); }}>Create an account</button>
                  </>
                )}
              </div>
            </motion.div>

            <div className="mt-5 flex items-start gap-2 text-[11px] text-muted-foreground justify-center">
              <ShieldCheck className="size-3.5 shrink-0 mt-0.5 text-emerald-600 dark:text-emerald-400" />
              <span>Your account lives on this device. Passwords are hashed and never leave your machine.</span>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
