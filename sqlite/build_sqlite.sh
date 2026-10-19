#!/bin/sh
# Download the official SQLite 3.46.1 amalgamation (public domain), check its SHA-256, and build it twice:
# with statement scan status (VM step counters) and, for the STAT4 control, additionally with STAT4 statistics.
# Works on macOS (.dylib) and Linux (.so).
set -eu
cd "$(dirname "$0")/vendor"
ZIP=sqlite-amalgamation-3460100.zip
SHA=77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784
if [ ! -f sqlite3.c ]; then
  [ -f $ZIP ] || curl -fsSLO https://www.sqlite.org/2024/$ZIP
  if command -v sha256sum >/dev/null; then GOT=$(sha256sum $ZIP | cut -d' ' -f1); else GOT=$(shasum -a 256 $ZIP | cut -d' ' -f1); fi
  [ "$GOT" = "$SHA" ] || { echo "checksum mismatch for $ZIP" >&2; exit 1; }
  unzip -j -o $ZIP 'sqlite-amalgamation-3460100/sqlite3.c' 'sqlite-amalgamation-3460100/sqlite3.h'
fi
case "$(uname)" in
  Darwin) EXT=dylib; SHARED=-dynamiclib ;;
  *)      EXT=so;    SHARED=-shared ;;
esac
CC=${CC:-cc}
$CC -O2 -fPIC $SHARED -DSQLITE_ENABLE_STMT_SCANSTATUS -DSQLITE_THREADSAFE=0 sqlite3.c -o libsqlite_profile.$EXT -lm
$CC -O2 -fPIC $SHARED -DSQLITE_ENABLE_STMT_SCANSTATUS -DSQLITE_ENABLE_STAT4 -DSQLITE_THREADSAFE=0 sqlite3.c -o libsqlite_stat4.$EXT -lm
echo "built vendor/libsqlite_profile.$EXT and vendor/libsqlite_stat4.$EXT"
