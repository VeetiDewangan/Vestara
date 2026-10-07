import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'wardrobe_item.dart';

/// The user's saved wardrobe. Persisted on the device with shared_preferences
/// so items are still there after the app restarts.
///
/// Widgets listen to it (ListenableBuilder) and rebuild on add/remove.
class WardrobeStore extends ChangeNotifier {
  WardrobeStore();

  /// App-wide instance. Tests create their own with `WardrobeStore()`.
  static final WardrobeStore instance = WardrobeStore();

  static const String _prefsKey = 'wardrobe_items_v1';

  final List<WardrobeItem> _items = [];

  /// Newest first.
  List<WardrobeItem> get items => List.unmodifiable(_items);

  Future<void> load() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_prefsKey);
      _items.clear();
      if (raw != null && raw.isNotEmpty) {
        final decoded = jsonDecode(raw);
        if (decoded is List) {
          for (final entry in decoded) {
            try {
              _items.add(WardrobeItem.fromJson(entry as Map<String, dynamic>));
            } catch (_) {
              // skip one corrupt entry instead of losing the whole wardrobe
            }
          }
        }
      }
    } catch (e) {
      debugPrint('Could not load wardrobe: $e');
    }
    notifyListeners();
  }

  Future<void> add(WardrobeItem item) async {
    _items.insert(0, item);
    notifyListeners();
    await _save();
  }

  Future<void> remove(String id) async {
    _items.removeWhere((i) => i.id == id);
    notifyListeners();
    await _save();
  }

  Future<void> _save() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(
        _prefsKey,
        jsonEncode(_items.map((i) => i.toJson()).toList()),
      );
    } catch (e) {
      // Keep the in-memory item; it just won't survive a restart.
      debugPrint('Could not save wardrobe: $e');
    }
  }
}
