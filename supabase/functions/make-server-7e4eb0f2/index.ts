import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "jsr:@supabase/supabase-js@2";
import * as kv from "./kv_store.ts";

const app = new Hono();
app.use("*", logger(console.log));
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

const supabaseAdmin = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
);

async function authedUserId(c: any): Promise<string | null> {
  const token = c.req.header("Authorization")?.split(" ")[1];
  if (!token) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(token);
  if (error || !data.user) return null;
  return data.user.id;
}

app.get("/make-server-7e4eb0f2/health", (c) => c.json({ status: "ok" }));

// Sign up
app.post("/make-server-7e4eb0f2/signup", async (c) => {
  try {
    const { email, password, name } = await c.req.json();
    if (!email || !password) return c.json({ error: "email and password required" }, 400);
    const { data, error } = await supabaseAdmin.auth.admin.createUser({
      email,
      password,
      user_metadata: { name: name || "" },
      // Auto-confirm since no email server is configured in this environment
      email_confirm: true,
    });
    if (error) {
      console.log(`Signup error for ${email}: ${error.message}`);
      return c.json({ error: error.message }, 400);
    }
    return c.json({ user: { id: data.user.id, email: data.user.email } });
  } catch (e) {
    console.log(`Signup unexpected error: ${e}`);
    return c.json({ error: `Signup failed: ${e}` }, 500);
  }
});

// List a user's saved sessions (metadata only)
app.get("/make-server-7e4eb0f2/sessions", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const items = await kv.getByPrefix(`session:${uid}:`);
    const meta = (items || []).map((s: any) => ({
      id: s.id,
      title: s.title,
      updated_at: s.updated_at,
      created_at: s.created_at,
    })).sort((a: any, b: any) => (b.updated_at || "").localeCompare(a.updated_at || ""));
    return c.json({ sessions: meta });
  } catch (e) {
    console.log(`List sessions error for ${uid}: ${e}`);
    return c.json({ error: `Failed to list sessions: ${e}` }, 500);
  }
});

// Load a single session
app.get("/make-server-7e4eb0f2/sessions/:id", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const id = c.req.param("id");
    const session = await kv.get(`session:${uid}:${id}`);
    if (!session) return c.json({ error: "Not found" }, 404);
    return c.json({ session });
  } catch (e) {
    console.log(`Load session error: ${e}`);
    return c.json({ error: `Failed to load session: ${e}` }, 500);
  }
});

// Save / update a session
app.put("/make-server-7e4eb0f2/sessions/:id", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const id = c.req.param("id");
    const body = await c.req.json();
    const now = new Date().toISOString();
    const existing = await kv.get(`session:${uid}:${id}`);
    const session = {
      id,
      title: body.title || existing?.title || "Untitled session",
      data: body.data ?? {},
      created_at: existing?.created_at || now,
      updated_at: now,
    };
    await kv.set(`session:${uid}:${id}`, session);
    return c.json({ session });
  } catch (e) {
    console.log(`Save session error: ${e}`);
    return c.json({ error: `Failed to save session: ${e}` }, 500);
  }
});

// Delete a session
app.delete("/make-server-7e4eb0f2/sessions/:id", async (c) => {
  const uid = await authedUserId(c);
  if (!uid) return c.json({ error: "Unauthorized" }, 401);
  try {
    const id = c.req.param("id");
    await kv.del(`session:${uid}:${id}`);
    return c.json({ ok: true });
  } catch (e) {
    console.log(`Delete session error: ${e}`);
    return c.json({ error: `Failed to delete session: ${e}` }, 500);
  }
});

Deno.serve(app.fetch);
