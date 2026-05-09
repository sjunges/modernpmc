/**
 * MyST transform plugin — adds a static GitHub "issue" link to every
 * labeled block (definitions, theorems, labeled paragraphs, admonitions).
 *
 * Runs at build time; no browser JS needed.
 * Register in myst.yml:
 *   project:
 *     plugins:
 *       - feedback-links.mjs
 */

const GITHUB_REPO = "sjunges/modernpmc";
const SITE_BASE   = "https://sjunges.github.io/modernpmc";

function discussionUrl(identifier, slug) {
  const htmlId  = identifier.replace(/[^a-z0-9_-]/gi, "-").toLowerCase();
  const pageUrl = `${SITE_BASE}/${slug}#${htmlId}`;
  const title   = encodeURIComponent(`Feedback: ${identifier}`);
  const body    = encodeURIComponent(
    `## Location\n\n[${pageUrl}](${pageUrl})\n\n## Feedback\n\nWrite your feedback here.\n`
  );
  return `https://github.com/${GITHUB_REPO}/discussions/new?category=general&title=${title}&body=${body}`;
}

export default {
  name: "Feedback Issue Links",
  transforms: [
    {
      name: "add-issue-links",
      stage: "document",
      doc: "Appends a GitHub issue link to section/subsection/subsubsection headings.",
      plugin: (_options, utils) => (node, vfile) => {
        const slug = (vfile?.path ?? "")
          .replace(/\\/g, "/").split("/").pop()
          ?.replace(/\.md$/, "") ?? "";

        // depth 2 = section, 3 = subsection, 4 = subsubsection; skip h1 (page title)
        const headings = utils.selectAll("heading", node)
          .filter((h) => h.depth >= 2 && h.identifier);

        for (const heading of headings) {
          heading.children.push({
            type: "link",
            url: discussionUrl(heading.identifier, slug),
            data: { class: "gh-issue-link" },
            children: [{ type: "text", value: "discuss!" }],
          });
        }
      },
    },
  ],
};