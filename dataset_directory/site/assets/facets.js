/**
 * The facet rail.
 *
 * Two of the facets are two-level trees. Neither taxonomy nor anatomy is a
 * strict hierarchy in this corpus — a species can occur under more than one
 * kingdom, and a tissue context under more than one organ — so the tree is
 * built from what actually co-occurs, and a child may appear under several
 * parents with a different count under each. Checking a parent is a shortcut,
 * not a claim about containment.
 */
import { NO_VALUE, TREE_FACETS } from './engine.js';

const COLLAPSED_LIMIT = 8;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function checkboxRow({ label, count, checked, indeterminate, disabled, onToggle }) {
  const row = element('label', 'facet-row' + (disabled ? ' is-empty' : ''));
  const box = element('input');
  box.type = 'checkbox';
  box.checked = checked;
  box.indeterminate = Boolean(indeterminate);
  box.disabled = disabled && !checked;
  box.addEventListener('change', onToggle);
  row.append(box, element('span', 'facet-label', label), element('span', 'facet-count', count.toLocaleString()));
  return row;
}

/**
 * A facet section that reveals its long tail on demand. Species and tissue
 * context have a hundred-plus values each; showing them all by default buries
 * every other facet below the fold.
 */
function section(title, rows, { expanded, onExpand }) {
  const wrapper = element('section', 'facet');
  wrapper.append(element('h3', null, title));
  const list = element('div', 'facet-list');
  const visible = expanded ? rows : rows.slice(0, COLLAPSED_LIMIT);
  visible.forEach((row) => list.append(row));
  wrapper.append(list);
  if (rows.length > COLLAPSED_LIMIT) {
    const more = element(
      'button',
      'facet-more',
      expanded ? 'Show fewer' : `Show all ${rows.length.toLocaleString()}`,
    );
    more.type = 'button';
    more.addEventListener('click', onExpand);
    wrapper.append(more);
  }
  if (!rows.length) wrapper.append(element('p', 'facet-empty', 'No values under the current filters'));
  return wrapper;
}

export function renderFacetRail({ facets, counts, selection, expandedKeys, actions }) {
  const rail = element('div', 'facet-rail');

  for (const facet of facets) {
    const selected = selection[facet.key];
    const tally = counts[facet.key] || new Map();
    const isExpanded = expandedKeys.has(facet.key);

    if (facet.kind === 'tree') {
      const { parent, child } = TREE_FACETS[facet.key];
      const rows = [];
      for (const root of facet.roots) {
        const parentKey = `${parent}:${root.id}`;
        const childIds = root.children.map((c) => c.id);
        const selectedChildren = childIds.filter((id) => selected.has(`${child}:${id}`));
        const parentChecked = selected.has(parentKey);
        const parentCount = tally.get(parentKey) || 0;

        const branch = element('div', 'facet-branch');
        const open = expandedKeys.has(`${facet.key}:${root.id}`);
        const disclosure = element('button', 'facet-disclosure', open ? '–' : '+');
        disclosure.type = 'button';
        disclosure.setAttribute('aria-expanded', String(open));
        disclosure.setAttribute('aria-label', `${open ? 'Collapse' : 'Expand'} ${root.label}`);
        disclosure.addEventListener('click', () => actions.toggleExpanded(`${facet.key}:${root.id}`));

        const head = element('div', 'facet-branch-head');
        head.append(
          disclosure,
          checkboxRow({
            label: root.label,
            count: parentCount,
            checked: parentChecked,
            indeterminate: !parentChecked && selectedChildren.length > 0,
            disabled: parentCount === 0,
            onToggle: () => actions.toggleTreeParent(facet.key, root.id, childIds),
          }),
        );
        branch.append(head);

        if (open) {
          const children = element('div', 'facet-children');
          for (const node of root.children) {
            const count = tally.get(`${child}:${node.id}`) || 0;
            children.append(
              checkboxRow({
                label: node.label,
                count,
                checked: selected.has(`${child}:${node.id}`),
                disabled: count === 0,
                onToggle: () => actions.toggle(facet.key, `${child}:${node.id}`),
              }),
            );
          }
          if (!root.children.length) {
            children.append(element('p', 'facet-empty', 'No tissue context recorded'));
          }
          branch.append(children);
        }
        rows.push(branch);
      }
      rail.append(
        section(facet.label, rows, {
          expanded: isExpanded,
          onExpand: () => actions.toggleExpanded(facet.key),
        }),
      );
      continue;
    }

    const rows = facet.values.map((value) =>
      checkboxRow({
        label: value.label,
        count: tally.get(value.id) || 0,
        checked: selected.has(value.id),
        disabled: (tally.get(value.id) || 0) === 0,
        onToggle: () => actions.toggle(facet.key, value.id),
      }),
    );
    if (facet.noValue) {
      // Assets missing this facet stay reachable rather than becoming
      // unselectable the moment anyone touches the facet.
      rows.push(
        checkboxRow({
          label: facet.noValue.label,
          count: tally.get(NO_VALUE) || 0,
          checked: selected.has(NO_VALUE),
          disabled: (tally.get(NO_VALUE) || 0) === 0,
          onToggle: () => actions.toggle(facet.key, NO_VALUE),
        }),
      );
    }
    rail.append(
      section(facet.label, rows, {
        expanded: isExpanded,
        onExpand: () => actions.toggleExpanded(facet.key),
      }),
    );
  }

  return rail;
}
