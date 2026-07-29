import type { ReactElement } from "react";

import logoSource from "../../Logo.png";

interface BrandLogoProps {
  className?: string;
  size?: "mark" | "hero";
}

export default function BrandLogo({
  className,
  size = "mark",
}: BrandLogoProps): ReactElement {
  return (
    <span
      className={["brand-logo", `brand-logo--${size}`, className]
        .filter(Boolean)
        .join(" ")}
      aria-hidden="true"
    >
      <img
        src={logoSource}
        alt=""
        width="500"
        height="500"
        draggable={false}
      />
    </span>
  );
}
