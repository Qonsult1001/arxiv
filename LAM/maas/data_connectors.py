#!/usr/bin/env python3
"""
🔌 DATA CONNECTORS - Connect Your Data Sources to the Brain

Connectors for:
- 📧 Email (Gmail, Outlook)
- 📁 Local Files
- 🔍 Browser History
- 📝 Notes (Obsidian, Notion)
- 💬 Chat History

Usage:
    from autonomous_brain import AutonomousBrain
    from data_connectors import EmailConnector, FileConnector, BrowserConnector
    
    brain = AutonomousBrain.load("willie.said")
    
    # Index local files
    FileConnector(brain).index_folder("~/Documents")
    
    # Connect email
    EmailConnector(brain).connect_gmail("you@gmail.com", "app_password")
    
    brain.save("willie.said")
"""

import os
import sys
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import mimetypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FileConnector:
    """
    📁 Connect local files to your brain.
    
    Indexes:
    - PDF documents
    - Text files (.txt, .md)
    - Code files (.py, .js, etc.)
    - Office documents (with optional support)
    """
    
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.rst', '.py', '.js', '.ts', '.html', '.css',
        '.json', '.yaml', '.yml', '.xml', '.csv', '.pdf',
    }
    
    def __init__(self, brain):
        self.brain = brain
        self.indexed_files: Dict[str, str] = {}  # path -> hash
    
    def index_folder(
        self, 
        folder_path: str, 
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        skip_hidden: bool = True
    ) -> Dict:
        """
        Index all files in a folder.
        
        Args:
            folder_path: Path to folder
            extensions: List of extensions to index (default: all supported)
            recursive: Index subdirectories
            skip_hidden: Skip hidden files/folders
            
        Returns:
            Summary of indexed files
        """
        folder_path = os.path.expanduser(folder_path)
        extensions = set(extensions or self.SUPPORTED_EXTENSIONS)
        
        indexed = 0
        skipped = 0
        errors = 0
        
        print(f"📁 Indexing folder: {folder_path}")
        
        if recursive:
            walker = os.walk(folder_path)
        else:
            walker = [(folder_path, [], os.listdir(folder_path))]
        
        for root, dirs, files in walker:
            # Skip hidden directories
            if skip_hidden:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                # Skip hidden files
                if skip_hidden and filename.startswith('.'):
                    continue
                
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    skipped += 1
                    continue
                
                filepath = os.path.join(root, filename)
                
                try:
                    # Check if already indexed (by hash)
                    file_hash = self._file_hash(filepath)
                    if filepath in self.indexed_files:
                        if self.indexed_files[filepath] == file_hash:
                            skipped += 1
                            continue
                    
                    # Index file
                    self._index_file(filepath)
                    self.indexed_files[filepath] = file_hash
                    indexed += 1
                    
                    if indexed % 10 == 0:
                        print(f"   ✅ Indexed {indexed} files...")
                        
                except Exception as e:
                    errors += 1
                    print(f"   ❌ Error indexing {filepath}: {e}")
        
        result = {
            "folder": folder_path,
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "total_known": len(self.indexed_files)
        }
        
        print(f"   ✅ Done! Indexed {indexed} files, skipped {skipped}, errors {errors}")
        
        return result
    
    def _index_file(self, filepath: str):
        """Index a single file."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf':
            # Use brain's PDF learning
            self.brain.learn_document(filepath)
        else:
            # Read as text
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Teach to brain with source
                self.brain.teach(
                    content,
                    category="file",
                )
            except UnicodeDecodeError:
                # Try with latin-1
                with open(filepath, 'r', encoding='latin-1') as f:
                    content = f.read()
                self.brain.teach(content, category="file")
    
    def _file_hash(self, filepath: str) -> str:
        """Get hash of file for change detection."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def watch_folder(self, folder_path: str, interval: int = 60):
        """
        Watch a folder for changes and index new/modified files.
        
        Args:
            folder_path: Path to watch
            interval: Check interval in seconds
        """
        import time
        
        print(f"👁️ Watching folder: {folder_path}")
        
        while True:
            self.index_folder(folder_path)
            time.sleep(interval)


class EmailConnector:
    """
    📧 Connect email to your brain.
    
    Supports:
    - Gmail (IMAP)
    - Outlook/Office365 (IMAP)
    - Any IMAP server
    """
    
    def __init__(self, brain):
        self.brain = brain
    
    def connect_gmail(
        self, 
        email_address: str, 
        app_password: str,
        max_emails: int = 100,
        folders: List[str] = None
    ) -> Dict:
        """
        Connect to Gmail and learn from emails.
        
        Args:
            email_address: Your Gmail address
            app_password: Gmail app password (not regular password!)
            max_emails: Maximum emails to fetch per folder
            folders: Folders to scan (default: INBOX, Sent)
            
        Returns:
            Summary of indexed emails
        """
        import imaplib
        import email
        from email.header import decode_header
        
        folders = folders or ['INBOX', '[Gmail]/Sent Mail']
        
        print(f"📧 Connecting to Gmail: {email_address}")
        
        try:
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(email_address, app_password)
        except Exception as e:
            return {"error": f"Failed to connect: {e}"}
        
        total_indexed = 0
        
        for folder in folders:
            try:
                mail.select(folder)
                _, messages = mail.search(None, 'ALL')
                
                message_ids = messages[0].split()[-max_emails:]
                print(f"   📂 {folder}: {len(message_ids)} emails")
                
                for num in message_ids:
                    try:
                        _, msg = mail.fetch(num, '(RFC822)')
                        email_msg = email.message_from_bytes(msg[0][1])
                        
                        # Extract email parts
                        subject = self._decode_header(email_msg['subject'])
                        sender = self._decode_header(email_msg['from'])
                        to = self._decode_header(email_msg['to'])
                        date = email_msg['date']
                        body = self._get_email_body(email_msg)
                        
                        # Create content for brain
                        content = f"""
Email from: {sender}
To: {to}
Date: {date}
Subject: {subject}

{body[:5000]}
                        """.strip()
                        
                        # Teach to brain
                        self.brain.teach(content, category="email")
                        total_indexed += 1
                        
                    except Exception as e:
                        print(f"   ⚠️ Error parsing email: {e}")
                        
            except Exception as e:
                print(f"   ⚠️ Error accessing folder {folder}: {e}")
        
        mail.logout()
        
        print(f"   ✅ Indexed {total_indexed} emails")
        
        return {"indexed": total_indexed, "folders": folders}
    
    def connect_outlook(
        self,
        email_address: str,
        password: str,
        max_emails: int = 100
    ) -> Dict:
        """
        Connect to Outlook/Office365.
        
        Same as Gmail but different server.
        """
        import imaplib
        
        print(f"📧 Connecting to Outlook: {email_address}")
        
        try:
            mail = imaplib.IMAP4_SSL('outlook.office365.com')
            mail.login(email_address, password)
        except Exception as e:
            return {"error": f"Failed to connect: {e}"}
        
        # Use similar logic as Gmail
        # ... (similar implementation)
        
        return {"indexed": 0, "note": "Implement similar to Gmail"}
    
    def _decode_header(self, header):
        """Decode email header."""
        if header is None:
            return ""
        
        from email.header import decode_header
        
        decoded = decode_header(header)
        parts = []
        for part, encoding in decoded:
            if isinstance(part, bytes):
                parts.append(part.decode(encoding or 'utf-8', errors='ignore'))
            else:
                parts.append(part)
        return ' '.join(parts)
    
    def _get_email_body(self, msg):
        """Extract email body."""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        return part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        pass
        else:
            try:
                return msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                pass
        return ""


class BrowserConnector:
    """
    🔍 Connect browser history to your brain.
    
    Supports:
    - Chrome
    - Firefox
    - Brave
    - Edge
    """
    
    def __init__(self, brain):
        self.brain = brain
    
    def index_chrome_history(self, max_entries: int = 1000) -> Dict:
        """
        Index Chrome browsing history.
        
        Note: Chrome must be closed for this to work.
        """
        import shutil
        
        # Find Chrome history file
        possible_paths = [
            os.path.expanduser("~/.config/google-chrome/Default/History"),  # Linux
            os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History"),  # Mac
            os.path.expanduser("~/AppData/Local/Google/Chrome/User Data/Default/History"),  # Windows
        ]
        
        history_path = None
        for path in possible_paths:
            if os.path.exists(path):
                history_path = path
                break
        
        if not history_path:
            return {"error": "Chrome history file not found"}
        
        print(f"🔍 Indexing Chrome history...")
        
        # Copy to avoid lock
        temp_path = "/tmp/chrome_history_copy"
        shutil.copy2(history_path, temp_path)
        
        try:
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT url, title, visit_count, last_visit_time 
                FROM urls 
                ORDER BY last_visit_time DESC 
                LIMIT ?
            """, (max_entries,))
            
            indexed = 0
            for url, title, visits, last_visit in cursor.fetchall():
                content = f"Visited website: {title}\nURL: {url}\nTimes visited: {visits}"
                self.brain.teach(content, category="browsing")
                indexed += 1
            
            conn.close()
            os.remove(temp_path)
            
            print(f"   ✅ Indexed {indexed} browsing entries")
            
            return {"indexed": indexed}
            
        except Exception as e:
            return {"error": str(e)}
    
    def index_firefox_history(self, max_entries: int = 1000) -> Dict:
        """Index Firefox browsing history."""
        import glob
        
        # Find Firefox profile
        firefox_path = os.path.expanduser("~/.mozilla/firefox/")
        profiles = glob.glob(os.path.join(firefox_path, "*.default*", "places.sqlite"))
        
        if not profiles:
            return {"error": "Firefox profile not found"}
        
        # Similar logic to Chrome...
        return {"note": "Implement similar to Chrome"}


class NotesConnector:
    """
    📝 Connect notes apps to your brain.
    
    Supports:
    - Obsidian (markdown vault)
    - Plain markdown folders
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.file_connector = FileConnector(brain)
    
    def index_obsidian_vault(self, vault_path: str) -> Dict:
        """Index an Obsidian vault."""
        print(f"📝 Indexing Obsidian vault: {vault_path}")
        
        return self.file_connector.index_folder(
            vault_path,
            extensions=['.md'],
            skip_hidden=True
        )


class ChatHistoryConnector:
    """
    💬 Import chat history from various sources.
    
    Supports:
    - ChatGPT export
    - Claude export
    - Custom JSON format
    """
    
    def __init__(self, brain):
        self.brain = brain
    
    def import_chatgpt_export(self, export_path: str) -> Dict:
        """
        Import ChatGPT conversation export.
        
        Get this from ChatGPT Settings > Data Controls > Export data
        """
        print(f"💬 Importing ChatGPT history: {export_path}")
        
        try:
            with open(export_path, 'r') as f:
                data = json.load(f)
        except:
            return {"error": "Failed to read export file"}
        
        indexed = 0
        
        for conversation in data.get('conversations', data):
            messages = []
            
            # Handle different export formats
            if isinstance(conversation, dict):
                for msg in conversation.get('mapping', {}).values():
                    if msg.get('message'):
                        role = msg['message'].get('author', {}).get('role', 'unknown')
                        content = msg['message'].get('content', {}).get('parts', [''])[0]
                        if content:
                            messages.append(f"{role}: {content}")
            
            if messages:
                conversation_text = "\n".join(messages)
                self.brain.teach(conversation_text, category="chatgpt_history")
                indexed += 1
        
        print(f"   ✅ Imported {indexed} conversations")
        
        return {"indexed": indexed}
    
    def import_cursor_history(self, history_path: str) -> Dict:
        """
        Import Cursor conversation history.
        
        Cursor stores history in its workspace.
        """
        print(f"💬 Importing Cursor history...")
        
        # Cursor stores in .cursor folder
        # Implementation depends on Cursor's format
        
        return {"note": "Check Cursor's history format"}


class AllConnectors:
    """
    🔌 Master connector - connect everything at once.
    
    Usage:
        connectors = AllConnectors(brain)
        connectors.connect_all(
            documents_folder="~/Documents",
            gmail_email="you@gmail.com",
            gmail_password="app_password",
            index_browser=True
        )
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.files = FileConnector(brain)
        self.email = EmailConnector(brain)
        self.browser = BrowserConnector(brain)
        self.notes = NotesConnector(brain)
        self.chat = ChatHistoryConnector(brain)
    
    def connect_all(
        self,
        documents_folder: Optional[str] = None,
        gmail_email: Optional[str] = None,
        gmail_password: Optional[str] = None,
        obsidian_vault: Optional[str] = None,
        index_browser: bool = False,
        chatgpt_export: Optional[str] = None
    ) -> Dict:
        """Connect all specified sources."""
        
        results = {}
        
        if documents_folder:
            results['documents'] = self.files.index_folder(documents_folder)
        
        if gmail_email and gmail_password:
            results['gmail'] = self.email.connect_gmail(gmail_email, gmail_password)
        
        if obsidian_vault:
            results['obsidian'] = self.notes.index_obsidian_vault(obsidian_vault)
        
        if index_browser:
            results['browser'] = self.browser.index_chrome_history()
        
        if chatgpt_export:
            results['chatgpt'] = self.chat.import_chatgpt_export(chatgpt_export)
        
        # Save brain
        print("💾 Saving brain...")
        self.brain.save(self.brain.name + ".said")
        
        return results


# ============================================================
# QUICK START
# ============================================================

def setup_personal_brain(
    brain_name: str = "willie",
    documents_folder: Optional[str] = None,
    **kwargs
):
    """
    Quick setup for personal brain.
    
    Example:
        brain = setup_personal_brain(
            brain_name="willie",
            documents_folder="~/Documents",
        )
    """
    from autonomous_brain import AutonomousBrain
    
    brain_file = f"{brain_name}.said"
    
    if os.path.exists(brain_file):
        brain = AutonomousBrain.load(brain_file)
    else:
        brain = AutonomousBrain(brain_name)
    
    connectors = AllConnectors(brain)
    
    results = connectors.connect_all(
        documents_folder=documents_folder,
        **kwargs
    )
    
    brain.save(brain_file)
    
    return brain, results


if __name__ == "__main__":
    print("=" * 60)
    print("🔌 DATA CONNECTORS TEST")
    print("=" * 60)
    
    from autonomous_brain import AutonomousBrain
    
    # Load or create brain
    if os.path.exists("willie.said"):
        brain = AutonomousBrain.load("willie.said")
    else:
        brain = AutonomousBrain("willie")
    
    # Test file connector
    print("\n📁 Testing File Connector...")
    files = FileConnector(brain)
    result = files.index_folder("test_documents/")
    print(f"   Result: {result}")
    
    # Save
    brain.save("willie.said")
    
    print("\n✅ Connectors tested!")

