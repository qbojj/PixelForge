# Kompilacja Pracy Dyplomowej

## Wymagania

### Linux/Mac
```bash
sudo apt-get install texlive-full texlive-lang-polish texlive-latex-extra
```

### Windows
Zainstaluj MiKTeX lub TeX Live z [https://www.latex-project.org/get/](https://www.latex-project.org/get/)

## Kompilacja

### Metoda 1: pdflatex (podstawowa)

```bash
cd thesis

# Pierwsza kompilacja (generuje pomocnicze pliki)
pdflatex thesis_new.tex

# Druga kompilacja (aktualizuje referencje)
pdflatex thesis_new.tex

# Trzecia kompilacja (finalizuje ToC i referencje)
pdflatex thesis_new.tex
```

### Metoda 2: latexmk (zalecana)

```bash
cd thesis

# Automatyczna wielokrotna kompilacja
latexmk -pdf thesis_new.tex

# Z ciągłym podglądem (aktualizuje przy zmianach)
latexmk -pdf -pvc thesis_new.tex

# Czyszczenie plików pomocniczych
latexmk -c
```

### Metoda 3: Visual Studio Code

1. Zainstaluj rozszerzenie "LaTeX Workshop"
2. Otwórz `thesis_new.tex`
3. Naciśnij Ctrl+Alt+B (Build) lub Ctrl+S (Auto-build)
4. Podgląd: Ctrl+Alt+V

## Struktura Plików

Po kompilacji pojawią się:

- `thesis_new.pdf` - **główny plik (TWOJA PRACA)**
- `thesis_new.aux` - pomocnicze (referencje)
- `thesis_new.log` - logi kompilacji
- `thesis_new.toc` - spis treści
- `thesis_new.out` - bookmarki PDF

## Sprawdzenie Poprawności

```bash
# Pokaż ostrzeżenia
grep -i warning thesis_new.log

# Pokaż błędy
grep -i error thesis_new.log

# Zlicz strony
pdfinfo thesis_new.pdf | grep Pages
```

## Edycja

### Dodawanie Rozdziałów

```latex
\chapter{Nowy Rozdział}

\section{Sekcja}

Treść...

\subsection{Podsekcja}

Więcej treści...
```

### Dodawanie Obrazków

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{obrazek.png}
\caption{Podpis obrazka}
\label{fig:etykieta}
\end{figure}
```

### Dodawanie Tabel

```latex
\begin{table}[H]
\centering
\caption{Tytuł tabeli}
\begin{tabular}{lcc}
\toprule
\textbf{Kolumna 1} & \textbf{Kolumna 2} & \textbf{Kolumna 3} \\
\midrule
Dane 1 & 123 & 456 \\
Dane 2 & 789 & 012 \\
\bottomrule
\end{tabular}
\end{table}
```

### Dodawanie Kodu

```latex
\begin{lstlisting}[language=Python, caption={Opis kodu}]
def funkcja():
    return "Hello World"
\end{lstlisting}
```

### Referencje

```latex
% Definicja
\label{sec:etykieta}

% Użycie
Jak pokazano w sekcji~\ref{sec:etykieta}...
```

## Typowe Problemy

### Problem: "File not found"
**Rozwiązanie**: Sprawdź czy plik `iithesis.cls` jest w tym samym katalogu

### Problem: Polskie znaki nie działają
**Rozwiązanie**: Upewnij się, że:
```latex
\usepackage[utf8]{inputenc}
\usepackage[polish]{babel}
\usepackage[T1]{fontenc}
```

### Problem: Bibliografia nie wyświetla się
**Rozwiązanie**: Użyj BibTeX:
```bash
pdflatex thesis_new.tex
bibtex thesis_new
pdflatex thesis_new.tex
pdflatex thesis_new.tex
```

## Wskazówki

1. **Często kompiluj** - łatwiej znajdziesz błędy
2. **Używaj \label i \ref** - automatyczne numerowanie
3. **Dodawaj \cite** - cytuj źródła
4. **Sprawdź marginesy** - muszą spełniać wymagania uczelni
5. **PDF/A** - sprawdź czy wymagany dla archiwum

## Wersjonowanie

```bash
# Zapisz backup przed dużymi zmianami
cp thesis_new.tex thesis_backup_$(date +%Y%m%d).tex

# Lub użyj git
git add thesis_new.tex
git commit -m "Rozdział 3 ukończony"
```

## Finalna Weryfikacja

Przed oddaniem sprawdź:

- [ ] Wszystkie rozdziały ukończone
- [ ] Spis treści poprawny
- [ ] Bibliografia kompletna
- [ ] Wszystkie obrazki i tabele mają podpisy
- [ ] Streszczenie PL i EN
- [ ] Dane na stronie tytułowej
- [ ] Numeracja stron poprawna
- [ ] Marginesy zgodne z wymaganiami
- [ ] Ortografia i gramatyka
- [ ] Format PDF/A (jeśli wymagany)

## Konwersja do PDF/A (jeśli wymagane)

```bash
# Ghostscript
gs -dPDFA=1 -dBATCH -dNOPAUSE -sProcessColorModel=DeviceRGB \
   -sDEVICE=pdfwrite -sPDFACompatibilityPolicy=1 \
   -sOutputFile=thesis_new_pdfa.pdf thesis_new.pdf
```

---

**Powodzenia z pracą dyplomową!** 🎓
