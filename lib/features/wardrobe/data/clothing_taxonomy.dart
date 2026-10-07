/// The category groups / subcategories shown in the Add Item screen, and how
/// the backend's AI categories map onto them.
///
/// Backend categories (backend/app/services/tagging.py):
///   t-shirt, shirt, blouse, top, sweater, hoodie, jacket, coat, dress, skirt,
///   jeans, trousers, shorts, leggings, kurta, saree, jumpsuit, other
class ClothingTaxonomy {
  ClothingTaxonomy._();

  static const List<String> groups = ['Bottom', 'Top', 'Outerwear', 'One-piece'];

  static const Map<String, List<String>> subcategories = {
    'Top': ['T-Shirt', 'Shirt', 'Blouse', 'Sweater', 'Hoodie', 'Top', 'Kurta'],
    'Bottom': ['Jeans', 'Trousers', 'Shorts', 'Skirt', 'Leggings'],
    'Outerwear': ['Jacket', 'Coat'],
    'One-piece': ['Dress', 'Saree', 'Jumpsuit'],
  };

  static const Map<String, ({String group, String sub})> _fromAi = {
    't-shirt': (group: 'Top', sub: 'T-Shirt'),
    'shirt': (group: 'Top', sub: 'Shirt'),
    'blouse': (group: 'Top', sub: 'Blouse'),
    'top': (group: 'Top', sub: 'Top'),
    'sweater': (group: 'Top', sub: 'Sweater'),
    'hoodie': (group: 'Top', sub: 'Hoodie'),
    'kurta': (group: 'Top', sub: 'Kurta'),
    'jacket': (group: 'Outerwear', sub: 'Jacket'),
    'coat': (group: 'Outerwear', sub: 'Coat'),
    'jeans': (group: 'Bottom', sub: 'Jeans'),
    'trousers': (group: 'Bottom', sub: 'Trousers'),
    'shorts': (group: 'Bottom', sub: 'Shorts'),
    'skirt': (group: 'Bottom', sub: 'Skirt'),
    'leggings': (group: 'Bottom', sub: 'Leggings'),
    'dress': (group: 'One-piece', sub: 'Dress'),
    'saree': (group: 'One-piece', sub: 'Saree'),
    'jumpsuit': (group: 'One-piece', sub: 'Jumpsuit'),
  };

  /// Returns null for `null`, "other" or anything unknown, so the screen keeps
  /// whatever the user has selected.
  static ({String group, String sub})? fromAiCategory(String? category) {
    if (category == null) return null;
    return _fromAi[category.trim().toLowerCase()];
  }
}
