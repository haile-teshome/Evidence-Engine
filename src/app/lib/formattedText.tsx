import { Fragment, ReactNode } from "react";

// PubMed titles and abstracts include a small set of inline formatting tags
// (italics for species names, superscripts for isotopes, etc.). This renderer
// converts a whitelisted subset into real React elements. Anything outside the
// whitelist is rendered as plain text, so it's XSS-safe.

const WHITELIST = ["i", "em", "sup", "sub", "b", "strong", "u"] as const;
type WhitelistTag = typeof WHITELIST[number];

const TAG_ALT = WHITELIST.join("|");
const TAG_RE = new RegExp(`<(${TAG_ALT})>([\\s\\S]*?)</\\1>`, "i");

export function FormattedText({ text, as = "span" }: { text: string; as?: keyof JSX.IntrinsicElements }) {
  const Wrapper = as as any;
  return <Wrapper>{render(text)}</Wrapper>;
}

function render(text: string, depth = 0): ReactNode {
  if (!text || depth > 4) return text;
  const out: ReactNode[] = [];
  let rest = text;
  let key = 0;
  while (rest) {
    const m = rest.match(TAG_RE);
    if (!m || m.index === undefined) {
      out.push(rest);
      break;
    }
    if (m.index > 0) out.push(<Fragment key={key++}>{rest.slice(0, m.index)}</Fragment>);
    const tag = m[1].toLowerCase() as WhitelistTag;
    const Tag = tag as any;
    out.push(<Tag key={key++}>{render(m[2], depth + 1)}</Tag>);
    rest = rest.slice(m.index + m[0].length);
  }
  return <>{out}</>;
}
