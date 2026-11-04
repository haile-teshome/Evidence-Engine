// A tab must show its grey placeholder while its content is on the way, then the
// content, with the placeholder gone once the crossfade has finished.
import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { DeferredPage } from "../DeferredPage";

describe("DeferredPage", () => {
  it("shows the skeleton first, then the page, then drops the skeleton", async () => {
    render(<DeferredPage><div>heavy page content</div></DeferredPage>);
    expect(screen.getByRole("status", { name: "Loading" })).toBeTruthy();
    expect(screen.queryByText("heavy page content")).toBeNull();
    expect(await screen.findByText("heavy page content")).toBeTruthy();
    await waitFor(() => expect(screen.queryByRole("status", { name: "Loading" })).toBeNull(), { timeout: 2000 });
  });
});
