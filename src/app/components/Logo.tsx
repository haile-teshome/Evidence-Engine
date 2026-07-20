// Brand mark for Evidence Engine: two overlapping italic serif "E"s on a
// deep-evergreen squircle — the same motif as the desktop app icon, so the app
// and its window read as one identity. The front E carries a thin knockout gap
// so both glyphs stay legible even at favicon size.

const SERIF = "Georgia, 'Times New Roman', serif";

export function LogoMark({ className = "size-9" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} role="img" aria-label="Evidence Engine">
      <defs>
        <linearGradient id="ee-tile" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#1a4b42" />
          <stop offset="1" stopColor="#0c302a" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="62" height="62" rx="17" fill="url(#ee-tile)" />
      <rect x="1" y="1" width="62" height="62" rx="17" fill="none" stroke="#ffffff" strokeOpacity="0.06" />
      {/* rear E — sage */}
      <text x="26" y="47" fontFamily={SERIF} fontStyle="italic" fontWeight="700" fontSize="46" textAnchor="middle" fill="#6fa894" fillOpacity="0.92">E</text>
      {/* knockout gap so the front E separates cleanly from the rear */}
      <text x="34" y="52" fontFamily={SERIF} fontStyle="italic" fontWeight="700" fontSize="46" textAnchor="middle" fill="none" stroke="#0c302a" strokeWidth="3">E</text>
      {/* front E — cream */}
      <text x="34" y="52" fontFamily={SERIF} fontStyle="italic" fontWeight="700" fontSize="46" textAnchor="middle" fill="#f4efe3">E</text>
    </svg>
  );
}

export function Logo({ markClassName = "size-9" }: { markClassName?: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark className={markClassName} />
      <div className="leading-none">
        <div className="text-[16px] tracking-tight" style={{ fontFamily: SERIF }}>
          <span className="font-bold text-foreground">Evidence</span>{" "}
          <span className="italic font-normal text-foreground/45">Engine</span>
        </div>
        <div className="mt-1.5 text-[9.5px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
          Systematic Reviews
        </div>
      </div>
    </div>
  );
}
