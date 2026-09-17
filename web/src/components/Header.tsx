import './Header.css';

interface HeaderProps {
  /** Which models answered, read from /health so it cannot drift from the
   * settings actually in force. Empty until that request lands. */
  modelLine: string;
}

export function Header({ modelLine }: HeaderProps) {
  return (
    <header className="app-header">
      <span className="app-header__title">Document Q&amp;A</span>
      <span className="app-header__spacer" />
      <span className="app-header__model">{modelLine}</span>
    </header>
  );
}
