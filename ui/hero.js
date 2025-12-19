/**
 * Hero Mode JavaScript - Chat Input Positioning
 * Dynamically centers chat input accounting for sidebar width
 */

function adjustChatInputPosition() {
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  const chatInput = document.querySelector(
    '[data-testid="stBottomBlockContainer"]'
  );

  if (!chatInput) return;

  // Get sidebar width (0 if hidden or collapsed)
  let sidebarWidth = 0;
  if (sidebar && window.getComputedStyle(sidebar).display !== "none") {
    sidebarWidth = sidebar.getBoundingClientRect().width;
  }

  // Calculate center of the content area
  const viewportWidth = window.innerWidth;
  const contentAreaWidth = viewportWidth - sidebarWidth;
  const contentCenterX = sidebarWidth + contentAreaWidth / 2;

  // Position the chat input centered in the content area
  const chatInputWidth = 700; // max-width
  const leftPosition = contentCenterX - chatInputWidth / 2;

  chatInput.style.left = leftPosition + "px";
  chatInput.style.transform = "none";
}

// Initialize hero mode positioning
function initHeroMode() {
  // Initial adjustment after DOM is ready
  setTimeout(adjustChatInputPosition, 100);

  // Adjust on window resize
  window.addEventListener("resize", adjustChatInputPosition);

  // Watch for sidebar state changes using MutationObserver
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  if (sidebar) {
    const observer = new MutationObserver(adjustChatInputPosition);
    observer.observe(sidebar, {
      attributes: true,
      attributeFilter: ["aria-expanded", "style"],
    });
  }
}

// Auto-initialize when script loads
initHeroMode();
