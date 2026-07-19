#!/usr/bin/env python3
"""Clear all data from the database for a fresh start."""

from src.rag_qa.core.database import get_db_session, DBSession, DBDocument, DBDocumentChunk, DBChatMessage
from sqlalchemy import text

print('🗑️  Clearing all data from database...')

db = get_db_session()
try:
    # Count existing data
    chunk_count = db.query(DBDocumentChunk).count()
    doc_count = db.query(DBDocument).count()
    session_count = db.query(DBSession).count()
    message_count = db.query(DBChatMessage).count()
    
    print(f'   Found {session_count} sessions, {doc_count} documents, {chunk_count} chunks, {message_count} messages')
    
    # Delete all data (order matters due to foreign keys)
    db.query(DBChatMessage).delete()
    print(f'   ✓ Deleted {message_count} chat messages')
    
    db.query(DBDocumentChunk).delete()
    print(f'   ✓ Deleted {chunk_count} chunks')
    
    db.query(DBDocument).delete()
    print(f'   ✓ Deleted {doc_count} documents')
    
    db.query(DBSession).delete()
    print(f'   ✓ Deleted {session_count} sessions')
    
    # Reset sequences (PostgreSQL)
    try:
        db.execute(text('TRUNCATE TABLE chat_messages RESTART IDENTITY CASCADE'))
        db.execute(text('TRUNCATE TABLE chunks RESTART IDENTITY CASCADE'))
        db.execute(text('TRUNCATE TABLE documents RESTART IDENTITY CASCADE'))
        db.execute(text('TRUNCATE TABLE sessions RESTART IDENTITY CASCADE'))
        print('   ✓ Reset ID sequences')
    except Exception as e:
        print(f'   ⚠️  Could not reset sequences: {e}')
    
    db.commit()
    print('\n✅ Database cleared successfully!')
    print('\n💡 Also clear browser localStorage for complete fresh start:')
    print('   1. Open browser DevTools (F12 or Cmd+Option+I)')
    print('   2. Go to Application/Storage tab')
    print('   3. Expand "Local Storage"')
    print('   4. Click on "http://localhost:3000"')
    print('   5. Right-click and select "Clear"')
    print('   6. Refresh the page (Cmd+R)')
    
except Exception as e:
    db.rollback()
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
finally:
    db.close()
