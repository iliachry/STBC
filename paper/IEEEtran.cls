%%
%% IEEEtran.cls - Minimal version for basic compilation
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2024/01/01 Minimal IEEEtran class]

% Load article as base class
\DeclareOption{twocolumn}{\PassOptionsToClass{twocolumn}{article}}
\DeclareOption{conference}{\PassOptionsToClass{10pt}{article}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}

% Setup two-column format
\RequirePackage{geometry}
\geometry{
    letterpaper,
    textwidth=7in,
    textheight=9.25in,
    columnsep=0.25in,
    top=0.75in,
    bottom=1in,
    left=0.75in,
    right=0.75in
}

% Define IEEEauthorblockN
\def\IEEEauthorblockN#1{#1}

% Define IEEEkeywords environment
\newenvironment{IEEEkeywords}{%
    \vspace{0.5em}
    \noindent\textbf{Index Terms---}%
}{}

% Adjust section formatting
\renewcommand\section{\@startsection{section}{1}{\z@}%
    {-3.5ex \@plus -1ex \@minus -.2ex}%
    {2.3ex \@plus.2ex}%
    {\normalfont\large\bfseries\centering}}

\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
    {-3.25ex\@plus -1ex \@minus -.2ex}%
    {1.5ex \@plus .2ex}%
    {\normalfont\normalsize\bfseries}}

% Adjust title formatting
\renewcommand\maketitle{%
    \twocolumn[%
        \begin{center}%
            {\LARGE\bfseries\@title\par}%
            \vspace{1em}%
            {\large\@author\par}%
            \vspace{0.5em}%
            {\@date\par}%
        \end{center}%
        \vspace{1em}%
    ]%
}

\endinput