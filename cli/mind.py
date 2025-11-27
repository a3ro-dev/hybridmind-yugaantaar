#!/usr/bin/env python3
"""
HybridMind .mind File CLI

Manage .mind database files from the command line.

Usage:
    python -m cli.mind info data/hybridmind.mind
    python -m cli.mind create mydb.mind
    python -m cli.mind export data/hybridmind.mind backup.mind.zip
    python -m cli.mind list data/
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.mindfile import MindFile, list_mind_files, MIND_EXTENSION


def cmd_info(args):
    """Show information about a .mind file."""
    mind = MindFile(args.path)
    
    if not mind.exists:
        print(f"❌ Not found: {args.path}")
        return 1
    
    info = mind.get_info()
    
    print(f"\n{'═' * 50}")
    print(f"  📁 {info['name']}{MIND_EXTENSION}")
    print(f"{'═' * 50}")
    print(f"  Path:     {info['path']}")
    print(f"  Version:  {info['version']}")
    print(f"  Created:  {info['created']}")
    print(f"  Modified: {info['modified']}")
    print(f"  Size:     {info['size_human']}")
    print()
    print(f"  📊 Statistics:")
    stats = info.get('stats', {})
    print(f"     Nodes:   {stats.get('nodes', 0):,}")
    print(f"     Edges:   {stats.get('edges', 0):,}")
    print(f"     Vectors: {stats.get('vectors', 0):,}")
    print()
    print(f"  💾 Components:")
    for name, size in info.get('component_sizes', {}).items():
        print(f"     {name}: {size:,} bytes")
    print(f"{'═' * 50}\n")
    
    return 0


def cmd_create(args):
    """Create a new .mind database."""
    mind = MindFile(args.path)
    
    if mind.exists:
        print(f"❌ Already exists: {mind.path}")
        return 1
    
    metadata = {}
    if args.description:
        metadata["description"] = args.description
    if args.author:
        metadata["author"] = args.author
    
    if mind.initialize(metadata=metadata):
        print(f"✅ Created: {mind.path}")
        return 0
    else:
        print(f"❌ Failed to create: {mind.path}")
        return 1


def cmd_export(args):
    """Export a .mind database to an archive."""
    mind = MindFile(args.path)
    
    if not mind.exists:
        print(f"❌ Not found: {args.path}")
        return 1
    
    output = args.output or f"{mind.name}_export"
    result = mind.export(output, compress=not args.no_compress)
    
    if result:
        print(f"✅ Exported to: {result}")
        return 0
    else:
        print(f"❌ Export failed")
        return 1


def cmd_import(args):
    """Import a .mind database from an archive."""
    result = MindFile.import_from(args.archive, args.target)
    
    if result:
        print(f"✅ Imported to: {result.path}")
        return 0
    else:
        print(f"❌ Import failed")
        return 1


def cmd_list(args):
    """List .mind files in a directory."""
    directory = args.directory or "."
    
    files = list_mind_files(directory)
    
    if not files:
        print(f"No .mind files found in: {directory}")
        return 0
    
    print(f"\n📂 .mind files in {directory}:\n")
    print(f"{'Name':<30} {'Size':<10} {'Nodes':<10} {'Edges':<10}")
    print("-" * 60)
    
    for f in files:
        name = f['name'] + MIND_EXTENSION
        size = f.get('size_human', '?')
        nodes = f.get('stats', {}).get('nodes', '?')
        edges = f.get('stats', {}).get('edges', '?')
        print(f"{name:<30} {size:<10} {nodes:<10} {edges:<10}")
    
    print()
    return 0


def cmd_delete(args):
    """Delete a .mind database."""
    mind = MindFile(args.path)
    
    if not mind.exists:
        print(f"❌ Not found: {args.path}")
        return 1
    
    if not args.force:
        response = input(f"Delete {mind.path}? This cannot be undone. (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    if mind.delete():
        print(f"✅ Deleted: {args.path}")
        return 0
    else:
        print(f"❌ Delete failed")
        return 1


def cmd_manifest(args):
    """Show the manifest.json contents."""
    mind = MindFile(args.path)
    
    if not mind.exists:
        print(f"❌ Not found: {args.path}")
        return 1
    
    manifest = mind.read_manifest()
    if manifest:
        print(json.dumps(manifest, indent=2))
        return 0
    else:
        print(f"❌ Could not read manifest")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="HybridMind .mind File Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info data/hybridmind.mind      Show database info
  %(prog)s create myknowledge.mind        Create new database
  %(prog)s export db.mind backup.zip      Export to archive
  %(prog)s list data/                     List all .mind files
  %(prog)s manifest data/hybridmind.mind  Show manifest.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # info
    p_info = subparsers.add_parser("info", help="Show .mind file information")
    p_info.add_argument("path", help="Path to .mind file")
    p_info.set_defaults(func=cmd_info)
    
    # create
    p_create = subparsers.add_parser("create", help="Create new .mind database")
    p_create.add_argument("path", help="Path for new .mind file")
    p_create.add_argument("-d", "--description", help="Description")
    p_create.add_argument("-a", "--author", help="Author name")
    p_create.set_defaults(func=cmd_create)
    
    # export
    p_export = subparsers.add_parser("export", help="Export to archive")
    p_export.add_argument("path", help="Path to .mind file")
    p_export.add_argument("-o", "--output", help="Output path")
    p_export.add_argument("--no-compress", action="store_true", help="Don't compress")
    p_export.set_defaults(func=cmd_export)
    
    # import
    p_import = subparsers.add_parser("import", help="Import from archive")
    p_import.add_argument("archive", help="Path to archive")
    p_import.add_argument("target", help="Target directory")
    p_import.set_defaults(func=cmd_import)
    
    # list
    p_list = subparsers.add_parser("list", help="List .mind files")
    p_list.add_argument("directory", nargs="?", default=".", help="Directory to scan")
    p_list.set_defaults(func=cmd_list)
    
    # delete
    p_delete = subparsers.add_parser("delete", help="Delete .mind database")
    p_delete.add_argument("path", help="Path to .mind file")
    p_delete.add_argument("-f", "--force", action="store_true", help="Skip confirmation")
    p_delete.set_defaults(func=cmd_delete)
    
    # manifest
    p_manifest = subparsers.add_parser("manifest", help="Show manifest.json")
    p_manifest.add_argument("path", help="Path to .mind file")
    p_manifest.set_defaults(func=cmd_manifest)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

