"""
webapp/api_meanings.py

Plain-English glosses for Windows API calls, used only to make the semantic explanation
readable ("this call does X"). They are a human aid for the analyst. The model never sees
them: what the semantic branch reads is the MPNet embedding of the call name.

Lookup order: exact name -> name without A/W/Ex suffix -> pattern rule -> CAPE category.
An unknown call is described by its category only, never guessed.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# name -> (capability, what the call does)
_EXACT: Dict[str, Tuple[str, str]] = {
    # registry
    "NtOpenKey": ("registry", "opens a registry key"),
    "NtCreateKey": ("registry", "creates or opens a registry key"),
    "NtQueryValueKey": ("registry", "reads a value from a registry key"),
    "NtSetValueKey": ("registry", "writes a value into a registry key"),
    "NtDeleteValueKey": ("registry", "deletes a value from a registry key"),
    "NtDeleteKey": ("registry", "deletes a registry key"),
    "NtEnumerateKey": ("registry", "lists the subkeys of a registry key"),
    "NtEnumerateValueKey": ("registry", "lists the values stored in a registry key"),
    "NtQueryKey": ("registry", "reads information about a registry key"),
    "NtQueryMultipleValueKey": ("registry", "reads several registry values at once"),
    "RegOpenKey": ("registry", "opens a registry key"),
    "RegCreateKey": ("registry", "creates or opens a registry key"),
    "RegQueryValue": ("registry", "reads a value from a registry key"),
    "RegSetValue": ("registry", "writes a value into a registry key"),
    "RegDeleteValue": ("registry", "deletes a value from a registry key"),
    "RegDeleteKey": ("registry", "deletes a registry key"),
    "RegEnumKey": ("registry", "lists the subkeys of a registry key"),
    "RegEnumValue": ("registry", "lists the values stored in a registry key"),
    "RegQueryInfoKey": ("registry", "reads information about a registry key"),
    "RegCloseKey": ("registry", "closes a registry key"),
    # file system
    "NtCreateFile": ("file system", "creates or opens a file"),
    "NtOpenFile": ("file system", "opens an existing file"),
    "NtReadFile": ("file system", "reads data from a file"),
    "NtWriteFile": ("file system", "writes data to a file"),
    "NtDeleteFile": ("file system", "deletes a file"),
    "NtQueryInformationFile": ("file system", "reads a file's metadata (size, position, attributes)"),
    "NtSetInformationFile": ("file system", "changes a file's metadata, e.g. renames or marks it for deletion"),
    "NtQueryAttributesFile": ("file system", "reads a file's attributes"),
    "NtQueryFullAttributesFile": ("file system", "reads a file's full attributes"),
    "NtQueryDirectoryFile": ("file system", "lists the contents of a directory"),
    "NtDeviceIoControlFile": ("file system", "sends a control command to a device driver"),
    "CopyFile": ("file system", "copies a file"),
    "MoveFileWithProgress": ("file system", "moves or renames a file"),
    "DeleteFile": ("file system", "deletes a file"),
    "FindFirstFile": ("file system", "starts searching a directory for files"),
    "FindNextFile": ("file system", "continues a directory search, file by file"),
    "GetFileAttributes": ("file system", "reads a file's attributes"),
    "SetFileAttributes": ("file system", "changes a file's attributes"),
    "GetFileSize": ("file system", "reads the size of a file"),
    "SetFilePointer": ("file system", "moves the read/write position inside a file"),
    "GetFileType": ("file system", "checks what kind of object a handle refers to"),
    "SetEndOfFile": ("file system", "truncates or extends a file"),
    "CreateDirectory": ("file system", "creates a directory"),
    "RemoveDirectory": ("file system", "deletes a directory"),
    "GetTempPath": ("file system", "looks up the temporary-files folder"),
    "GetSystemDirectory": ("file system", "looks up the Windows system folder"),
    "GetVolumeNameForVolumeMountPoint": ("system reconnaissance", "looks up the identity of a disk volume"),
    "GetDiskFreeSpace": ("system reconnaissance", "checks free space on a disk"),
    # process, thread and memory
    "NtTerminateProcess": ("process control", "ends a process"),
    "NtCreateUserProcess": ("process control", "starts a new process"),
    "CreateProcessInternal": ("process control", "starts a new process"),
    "NtOpenProcess": ("process control", "opens a handle to another running process"),
    "NtCreateThreadEx": ("process control", "starts a new thread, possibly in another process"),
    "NtOpenThread": ("process control", "opens a handle to a thread"),
    "NtResumeThread": ("process control", "resumes a suspended thread"),
    "NtSuspendThread": ("process control", "pauses a thread"),
    "NtGetContextThread": ("process control", "reads a thread's CPU registers"),
    "NtSetContextThread": ("process control", "overwrites a thread's CPU registers"),
    "NtQueueApcThread": ("process control", "queues code to run inside a thread"),
    "ShellExecuteEx": ("process control", "launches a program or document"),
    "NtAllocateVirtualMemory": ("memory", "reserves or commits memory"),
    "NtFreeVirtualMemory": ("memory", "releases memory"),
    "NtProtectVirtualMemory": ("memory", "changes memory protection, e.g. makes memory executable"),
    "NtReadVirtualMemory": ("memory", "reads memory of a process"),
    "ReadProcessMemory": ("memory", "reads the memory of another process"),
    "NtWriteVirtualMemory": ("memory", "writes into the memory of a process"),
    "WriteProcessMemory": ("memory", "writes into the memory of another process"),
    "NtMapViewOfSection": ("memory", "maps a shared memory section into a process"),
    "NtUnmapViewOfSection": ("memory", "unmaps a memory section from a process"),
    "NtCreateSection": ("memory", "creates a shared memory section"),
    "NtOpenSection": ("memory", "opens a shared memory section"),
    "VirtualProtectEx": ("memory", "changes memory protection in a process"),
    # code loading
    "LdrLoadDll": ("code loading", "loads a DLL into the process"),
    "LdrGetDllHandle": ("code loading", "looks up a DLL already loaded in the process"),
    "LdrGetProcedureAddress": ("code loading", "resolves a function address at run time, a common way to hide which functions are used"),
    "LdrUnloadDll": ("code loading", "unloads a DLL"),
    "LoadLibrary": ("code loading", "loads a DLL into the process"),
    "GetProcAddress": ("code loading", "resolves a function address at run time"),
    # cryptography
    "CryptAcquireContext": ("cryptography", "opens a cryptographic provider"),
    "CryptCreateHash": ("cryptography", "creates a hash object"),
    "CryptHashData": ("cryptography", "feeds data into a hash"),
    "CryptDestroyHash": ("cryptography", "releases a hash object"),
    "CryptEncrypt": ("cryptography", "encrypts data"),
    "CryptDecrypt": ("cryptography", "decrypts data"),
    "CryptGenKey": ("cryptography", "generates a cryptographic key"),
    "CryptImportKey": ("cryptography", "imports a cryptographic key"),
    "CryptExportKey": ("cryptography", "exports a cryptographic key"),
    "CryptGenRandom": ("cryptography", "generates random bytes"),
    "CryptDeriveKey": ("cryptography", "derives a key from a password or hash"),
    # network
    "socket": ("network", "creates a network socket"),
    "connect": ("network", "connects to a remote host"),
    "send": ("network", "sends data over the network"),
    "recv": ("network", "receives data from the network"),
    "sendto": ("network", "sends data to a network address"),
    "recvfrom": ("network", "receives data from a network address"),
    "bind": ("network", "binds a socket to a local address"),
    "listen": ("network", "waits for incoming connections"),
    "setsockopt": ("network", "configures a network socket"),
    "closesocket": ("network", "closes a network socket"),
    "getaddrinfo": ("network", "resolves a host name to an address"),
    "gethostbyname": ("network", "resolves a host name to an address"),
    "DnsQuery_": ("network", "sends a DNS query"),
    "InternetOpen": ("network", "starts an internet session"),
    "InternetConnect": ("network", "connects to an internet server"),
    "InternetOpenUrl": ("network", "opens a URL"),
    "InternetReadFile": ("network", "downloads data from an internet connection"),
    "HttpOpenRequest": ("network", "prepares an HTTP request"),
    "HttpSendRequest": ("network", "sends an HTTP request"),
    "HttpAddRequestHeaders": ("network", "adds headers to an HTTP request"),
    "URLDownloadToFile": ("network", "downloads a file from a URL"),
    "WSAStartup": ("network", "initialises Windows networking"),
    # system reconnaissance
    "NtQuerySystemInformation": ("system reconnaissance", "queries system-wide information, e.g. running processes"),
    "GetSystemInfo": ("system reconnaissance", "reads hardware information such as CPU count"),
    "GetNativeSystemInfo": ("system reconnaissance", "reads hardware information such as CPU count"),
    "GetSystemMetrics": ("system reconnaissance", "reads display and system settings"),
    "GetComputerName": ("system reconnaissance", "reads the computer's name"),
    "GetUserName": ("system reconnaissance", "reads the current user's name"),
    "GetVersionEx": ("system reconnaissance", "reads the Windows version"),
    "RtlGetVersion": ("system reconnaissance", "reads the Windows version"),
    "IsDebuggerPresent": ("system reconnaissance", "checks whether a debugger is attached"),
    "NtQueryInformationProcess": ("system reconnaissance", "reads information about a process"),
    "Process32FirstW": ("system reconnaissance", "starts listing running processes"),
    "Process32NextW": ("system reconnaissance", "continues listing running processes"),
    "CreateToolhelp32Snapshot": ("system reconnaissance", "takes a snapshot of running processes"),
    "GetAdaptersAddresses": ("system reconnaissance", "lists network adapters"),
    "GlobalMemoryStatusEx": ("system reconnaissance", "reads how much memory the machine has"),
    "FindWindow": ("system reconnaissance", "looks for a window by name, often to detect tools"),
    "EnumWindows": ("system reconnaissance", "lists open windows"),
    "GetForegroundWindow": ("system reconnaissance", "checks which window is in front"),
    "GetCursorPos": ("system reconnaissance", "reads the mouse position"),
    "SystemParametersInfo": ("user interface", "reads or changes system-wide display and desktop settings"),
    "GetKeyState": ("system reconnaissance", "reads the state of a key"),
    "GetAsyncKeyState": ("system reconnaissance", "reads key presses as they happen"),
    # time
    "GetSystemTime": ("time", "reads the current system time"),
    "GetSystemTimeAsFileTime": ("time", "reads the current system time"),
    "GetLocalTime": ("time", "reads the local time"),
    "NtQuerySystemTime": ("time", "reads the current system time"),
    "GetTickCount": ("time", "reads how long the system has been running"),
    "NtDelayExecution": ("time", "sleeps, i.e. pauses execution for a while"),
    "timeGetTime": ("time", "reads a millisecond timer"),
    # synchronisation and handles
    "NtClose": ("handles", "closes a handle to an object"),
    "NtDuplicateObject": ("handles", "duplicates a handle, possibly into another process"),
    "NtCreateMutant": ("synchronisation", "creates a mutex, often used to run only once per machine"),
    "NtOpenMutant": ("synchronisation", "opens an existing mutex"),
    "NtReleaseMutant": ("synchronisation", "releases a mutex"),
    "NtCreateEvent": ("synchronisation", "creates an event object"),
    "NtOpenEvent": ("synchronisation", "opens an event object"),
    "NtWaitForSingleObject": ("synchronisation", "waits for an object to be signalled"),
    "NtAddAtomEx": ("synchronisation", "adds a string to the global atom table"),
    "NtFindAtom": ("synchronisation", "looks up a string in the global atom table"),
    # services
    "OpenSCManager": ("services", "opens the service control manager"),
    "CreateService": ("services", "installs a Windows service"),
    "OpenService": ("services", "opens a Windows service"),
    "StartService": ("services", "starts a Windows service"),
    "ControlService": ("services", "sends a command to a Windows service"),
    "DeleteService": ("services", "removes a Windows service"),
    # misc
    "RtlSetCurrentTransaction": ("file system", "sets the file-system transaction the thread works in"),
    "SetErrorMode": ("error handling", "controls how errors are reported, e.g. to suppress dialogs"),
    "SetUnhandledExceptionFilter": ("error handling", "installs a handler for unhandled crashes"),
    "LdrGetProcedureAddressForCaller": ("code loading", "resolves a function address at run time"),
    "CoCreateInstance": ("COM", "creates a COM object"),
    "CoInitializeEx": ("COM", "initialises COM for the thread"),
    "OleInitialize": ("COM", "initialises OLE"),
}

# (regex on the name, capability, description) for call families not listed above
_RULES = [
    (r"^Crypt", "cryptography", "performs a cryptographic operation"),
    (r"^(Reg|NtQuery.*Key|NtSet.*Key)", "registry", "accesses the registry"),
    (r"^(Nt|Zw).*File", "file system", "performs a file operation"),
    (r"(Process|Thread)", "process control", "manages a process or thread"),
    (r"(Virtual|Section|Heap)", "memory", "manages memory"),
    (r"^(Internet|Http|WSA|Dns|Url|URL)", "network", "performs a network operation"),
    (r"(Window|Cursor|Key(board)?State|Message)", "user interface", "interacts with windows or input"),
    (r"(Service|SCManager)", "services", "manages Windows services"),
    (r"(Time|Tick|Delay)", "time", "reads or waits on time"),
    (r"(Mutant|Mutex|Event|Semaphore|Atom)", "synchronisation", "synchronises with other code"),
    (r"^(Ldr|LoadLibrary|GetProcAddress|GetModule)", "code loading", "loads or resolves code"),
]

_CATEGORY_GLOSS = {
    "registry": "registry", "filesystem": "file system", "file": "file system",
    "process": "process control", "threading": "process control", "memory": "memory",
    "network": "network", "crypto": "cryptography", "system": "system reconnaissance",
    "synchronisation": "synchronisation", "synchronization": "synchronisation",
    "services": "services", "ui": "user interface", "misc": "miscellaneous",
    "hooking": "hooking", "device": "device", "com": "COM", "browser": "network",
}


def _strip_suffix(name: str) -> str:
    for suffix in ("ExW", "ExA", "Ex", "W", "A"):
        if name.endswith(suffix) and name[: -len(suffix)] in _EXACT:
            return name[: -len(suffix)]
    return name


def describe(name: str, category: Optional[str] = None) -> Dict[str, str]:
    """Return {'capability', 'meaning', 'source'} for one API call name."""
    name = (name or "").strip()
    if name in _EXACT:
        cap, text = _EXACT[name]
        return {"capability": cap, "meaning": text, "source": "glossary"}
    base = _strip_suffix(name)
    if base in _EXACT:
        cap, text = _EXACT[base]
        return {"capability": cap, "meaning": text, "source": "glossary"}
    for pattern, cap, text in _RULES:
        if re.search(pattern, name):
            return {"capability": cap, "meaning": text, "source": "pattern"}
    cat = _CATEGORY_GLOSS.get((category or "").strip().lower())
    if cat:
        return {"capability": cat, "meaning": f"a {cat} call", "source": "category"}
    return {"capability": "other", "meaning": "no description available", "source": "none"}
