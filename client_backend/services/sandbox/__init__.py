"""A separate local Windows account for the commands the assistant runs.

Commands run as ``KaniSandbox`` instead of the signed-in user, so they cannot
read the user's profile (keys, browser data, the sidecar's own secrets) or
decrypt the user's DPAPI-protected data, and can write only where the account
has been granted access: the configured workspace roots.
"""
