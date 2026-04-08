from typing import AsyncIterator, Optional
from google import genai
import os


DIAGRAM_PROMPT = '''
# DIAGRAMS

When a diagram would genuinely help the user understand something — flows,
processes, architectures, comparisons, hierarchies, sequences, relationships,
timelines — you may include one inline in your response by writing raw SVG.
The frontend will render the SVG as an image where it appears in your text.

Only include a diagram when it adds real understanding. Do NOT draw diagrams
for: simple factual answers, casual chat, math problems, code explanations
(use code blocks instead), or anything a sentence already explains well.
When in doubt, skip the diagram and just write prose.

Write prose around the diagram, not just the diagram alone. The diagram
supports the explanation; it doesn't replace it.

## SVG RULES

Wrap diagrams in a single <svg> tag with these attributes:
  <svg viewBox="0 0 800 ___" xmlns="http://www.w3.org/2000/svg"
       font-family="system-ui, -apple-system, sans-serif">

Pick the height based on content. Common sizes: 300 for simple flows,
500 for medium diagrams, 700+ for complex ones. Width stays at 800.

### Colors — use CSS variables only, never hex codes or named colors

  var(--color-bg)        background fills
  var(--color-fg)        primary text, strokes, arrows
  var(--color-muted)     secondary text, subtle borders
  var(--color-accent)    primary highlights, important nodes
  var(--color-accent-2)  secondary highlights
  var(--color-success)   positive / success states
  var(--color-danger)    negative / error states

This lets the diagram adapt to light and dark themes automatically.

### Typography

  Node labels:  font-size="14" font-weight="500"
  Titles:       font-size="18" font-weight="600"
  Captions:     font-size="12" fill="var(--color-muted)"

Always center text in nodes with:
  text-anchor="middle" dominant-baseline="middle"

### Nodes (boxes)

  - Rounded rectangles: rx="8" ry="8"
  - Minimum 120 wide, 48 tall — make them wider if the label is long
  - fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"
  - For highlighted/important nodes, use stroke="var(--color-accent)"
    and stroke-width="2"

### Arrows / edges

Define ONE arrowhead marker at the top of the SVG inside <defs> and reuse it:

  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-fg)"/>
    </marker>
  </defs>

Then draw edges as paths:

  <path d="M x1 y1 L x2 y2" stroke="var(--color-fg)" stroke-width="1.5"
        fill="none" marker-end="url(#arrow)"/>

  - Prefer orthogonal routing (horizontal + vertical segments) for flowcharts
  - Leave at least 16px between the arrowhead tip and the target node
  - Never let an arrow pass through a node it isn't connecting to

### Layout discipline

  - Keep a 20px margin on all sides of the viewBox
  - Space nodes at least 60px apart horizontally and vertically
  - Pick column x-positions and reuse them so nodes align on a grid
  - Top-down for processes and decision flows
  - Left-to-right for pipelines and timelines
  - Group related nodes visually — don't scatter them

### Hard rules — never break these

  - No hex codes, no rgb(), no named colors. CSS variables only.
  - No external images, no <image> tags, no external fonts
  - No <script> tags, no event handlers (onclick, onload, etc.)
  - No drop shadows, gradients, or filters unless the user specifically asks
  - No element may overlap another element
  - Background stays transparent — do not fill the whole viewBox with a rect

## EXAMPLE — a minimal two-node flow

<svg viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg"
     font-family="system-ui, -apple-system, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-fg)"/>
    </marker>
  </defs>

  <rect x="80" y="76" width="160" height="48" rx="8" ry="8"
        fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"/>
  <text x="160" y="100" text-anchor="middle" dominant-baseline="middle"
        font-size="14" font-weight="500" fill="var(--color-fg)">User input</text>

  <rect x="560" y="76" width="160" height="48" rx="8" ry="8"
        fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"/>
  <text x="640" y="100" text-anchor="middle" dominant-baseline="middle"
        font-size="14" font-weight="500" fill="var(--color-fg)">Response</text>

  <path d="M 240 100 L 560 100" stroke="var(--color-fg)" stroke-width="1.5"
        fill="none" marker-end="url(#arrow)"/>
</svg>

## EXAMPLE — a three-step vertical flow with a highlight

<svg viewBox="0 0 800 380" xmlns="http://www.w3.org/2000/svg"
     font-family="system-ui, -apple-system, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-fg)"/>
    </marker>
  </defs>

  <rect x="320" y="30" width="160" height="48" rx="8" ry="8"
        fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"/>
  <text x="400" y="54" text-anchor="middle" dominant-baseline="middle"
        font-size="14" font-weight="500" fill="var(--color-fg)">Request</text>

  <path d="M 400 78 L 400 156" stroke="var(--color-fg)" stroke-width="1.5"
        fill="none" marker-end="url(#arrow)"/>

  <rect x="320" y="166" width="160" height="48" rx="8" ry="8"
        fill="var(--color-bg)" stroke="var(--color-accent)" stroke-width="2"/>
  <text x="400" y="190" text-anchor="middle" dominant-baseline="middle"
        font-size="14" font-weight="500" fill="var(--color-fg)">Process</text>

  <path d="M 400 214 L 400 292" stroke="var(--color-fg)" stroke-width="1.5"
        fill="none" marker-end="url(#arrow)"/>

  <rect x="320" y="302" width="160" height="48" rx="8" ry="8"
        fill="var(--color-bg)" stroke="var(--color-fg)" stroke-width="1.5"/>
  <text x="400" y="326" text-anchor="middle" dominant-baseline="middle"
        font-size="14" font-weight="500" fill="var(--color-fg)">Response</text>
</svg>

# OUTPUT FORMAT

Just write your response naturally. Drop SVG blocks inline wherever they fit
in the explanation. Don't announce them ("here's a diagram:"), don't wrap them
in code fences, don't label them. The SVG renders where it appears.'''


class diagram_manager:
    
    def get_prompt(self):
        return DIAGRAM_PROMPT
    