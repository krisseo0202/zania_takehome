import './FilterChips.css';

export type FilterId = 'all' | 'answered' | 'missing';

interface FilterChipsProps {
  active: FilterId;
  counts: Record<FilterId, number>;
  onChange: (id: FilterId) => void;
}

const CHIPS: { id: FilterId; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'answered', label: 'Answered' },
  { id: 'missing', label: 'Data Not Available' },
];

export function FilterChips({ active, counts, onChange }: FilterChipsProps) {
  return (
    <section className="filter-chips" aria-label="Filter results">
      {CHIPS.map((chip) => (
        <button
          key={chip.id}
          type="button"
          className={`filter-chip${chip.id === active ? ' filter-chip--active' : ''}`}
          aria-pressed={chip.id === active}
          onClick={() => onChange(chip.id)}
        >
          {chip.label}
          <span className="filter-chip__count">{counts[chip.id]}</span>
        </button>
      ))}
      <span className="filter-chips__spacer" />
      <span className="filter-chips__note">order preserved from the questions file</span>
    </section>
  );
}
