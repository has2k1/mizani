# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = mizani
SOURCEDIR     = .
PAPER         =
BUILDDIR      = _build

# For make theme
BOOTSWATCHTHEME = united
brand-primary = \#145FCE
headings-font-weight = bold
font-family-sans-serif = '"system-ui", "Segoe UI", "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"'

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "  theme    to create a custom sphinx boostrap theme"

.PHONY: help Makefile theme

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

theme:
	../tools/build_theme.sh --theme $(BOOTSWATCHTHEME) \
	   --brand-primary '$(brand-primary)' \
	   --headings-font-weight $(headings-font-weight) \
	   --font-family-sans-serif '$(font-family-sans-serif)'
