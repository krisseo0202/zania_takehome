import { formatSources } from '../format';
import './ResultsTable.css';

export interface ResultsRow {
  index: number;
  question: string;
  answer: string;
  isMissing: boolean;
  sources?: string[];
}

interface ResultsTableProps {
  rows: ResultsRow[];
}

/** A real <table> (not the div-grid the mockup uses) so the header/data
 * relationship is exposed to assistive tech. Sticky <thead> keeps the column
 * labels visible while <tbody> scrolls, matching "header stays". */
export function ResultsTable({ rows }: ResultsTableProps) {
  return (
    <div className="results-table-scroll">
      <table className="results-table">
        <colgroup>
          <col className="results-table__col-index" />
          <col className="results-table__col-question" />
          <col />
          <col className="results-table__col-source" />
        </colgroup>
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Question</th>
            <th scope="col">Answer</th>
            <th scope="col" className="results-table__source-head">
              Source
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.index}>
              <td className="results-table__index">{String(row.index).padStart(2, '0')}</td>
              <td className="results-table__question">{row.question}</td>
              <td className={row.isMissing ? 'results-table__answer results-table__answer--missing' : 'results-table__answer'}>
                {row.answer}
              </td>
              {/* Best-matching retrieved passage, with a count of the rest.
                  Evidence shown to the model, not a verified citation. */}
              <td className="results-table__source" title={row.sources?.join(', ')}>
                {formatSources(row.sources)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
