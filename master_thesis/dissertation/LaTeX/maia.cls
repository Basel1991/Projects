%%
%% This is file `maia.cls',
%% based on the class elsarticle.cls
%% (https://www.ctan.org/pkg/elsarticle)
%%
%% -------------------------------------------

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{maia}[2017/09/01 The MAIA master custom class]
\LoadClassWithOptions{elsarticle}
\RequirePackage{geometry}
\RequirePackage[table]{xcolor}
\RequirePackage{adjustbox}

%%some definitions
\newcommand{\MONTH}{%
  \ifcase\the\month
  \or January% 1
  \or February% 2
  \or March% 3
  \or April% 4
  \or May% 5
  \or June% 6
  \or July% 7
  \or August% 8
  \or September% 9
  \or October% 10
  \or November% 11
  \or December% 12
  \fi}

%%removing elsevier footnote
\renewcommand{\ps@pprintTitle}{%
     \let\@oddhead\@empty
     \let\@evenhead\@empty
     \let\@oddfoot\@empty
     \let\@evenfoot\@oddfoot}

%%heading of the first page
\renewcommand{\finalMaketitle}{%
  \resetTitleCounters
  \def\baselinestretch{1}%
   \MaiaBox
   \MaketitleBox
   \thispagestyle{pprintTitle}%
  \gdef\thefootnote{\arabic{footnote}}%
}

\def\MaiaBox{%
\vspace{-1cm}
\centering
\begin{tabular}{|p{2.5cm}|| p{10cm}||p{2.5cm}|}
  \hline
  \adjustbox{valign=c}{\includegraphics[width=2cm]{maia.eps}} &
  \adjustbox{valign=c}{{\cellcolor{gray!15}} \parbox[c]{9cm}{\centering \linespread{1.5} \LARGE Medical Imaging and Applications  \color{gray} Master Thesis, \MONTH~\the\year} } &
  \adjustbox{valign=c}{\includegraphics[width=2.5cm]{maia2.eps} }\\
  \hline
\end{tabular}
\vspace{1cm}
}

\geometry{%
  textheight=682pt,
  headsep=12pt
}

%%headers of subsequent pages
\usepackage{fancyhdr} % Required for header and footer configuration
\pagestyle{fancy}
\fancyhf{}
\fancyhead[RO]{\thepage} % Font setting for the page number in the header
\fancyhead[LO]{\@title} % Print the nearest section name on the left side of odd pages
%\fancyhead[RE]{\elsauthors} % Print the current chapter name on the right side of even pages
\renewcommand{\headrulewidth}{0.5pt} % Width of the rule under the header

