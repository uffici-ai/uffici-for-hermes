"""The local wire's contract, as shipped files.

🔴 A **byte-identical copy** of the companion's ``wire/v1/``, not an import.
The two children are standalone by construction — the plugin cannot read the
companion's tree, and each is exported to its own repository — so the contract
is duplicated deliberately and pinned by
``scripts/__tests__/uffici-wire-golden-parity.test.mjs`` in the platform
repository. Edit the companion's copy; the parity guard fails until this one
matches.
"""

from pathlib import Path

#: The contract's root, containing ``schema/``, ``golden/`` and ``MANIFEST.json``.
WIRE_V1 = Path(__file__).resolve().parent / "v1"

__all__ = ["WIRE_V1"]
