import 'package:flutter/material.dart';
import '../../../core/constants/app_colors.dart';
import '../../outfit_builder/screens/create_outfit_screen.dart';
import '../../wardrobe/data/wardrobe_item.dart';
import '../../wardrobe/data/wardrobe_store.dart';
import '../../wardrobe/screens/add_item_screen.dart';
import '../../wardrobe/widgets/wardrobe_image.dart';
import 'landing_page.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  int _currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
  backgroundColor: Colors.transparent,
  elevation: 0,
  leading: IconButton(
    icon: Container(
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: AppColors.primaryWine.withOpacity(0.12),
        shape: BoxShape.circle,
      ),
      child: const Icon(Icons.arrow_back, size: 18, color: AppColors.primaryWine),
    ),
    onPressed: () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const LandingPage()),
      );
    },
  ),
  title: const Text(
    'Outfits Calendar',
    style: TextStyle(
      fontSize: 22,
      fontWeight: FontWeight.bold,
      color: AppColors.textDark,
    ),
  ),
),
      
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildWeatherOutfitCard(),
              const SizedBox(height: 28),
              _buildSectionHeader('My Outfits'),
              const SizedBox(height: 12),
              _buildOutfitsList(),
              const SizedBox(height: 28),
              _buildSectionHeader('Items'),
              const SizedBox(height: 12),
              _buildItemsList(),
            ],
          ),
        ),
      ),
      bottomNavigationBar: _buildFloatingNavBar(),
    );
  }

  Widget _buildSectionHeader(String title) {
  return Row(
    children: [
      Text(
        title,
        style: const TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.bold,
          color: AppColors.textDark,
        ),
      ),
      const Spacer(),
      GestureDetector(
        onTap: () {
          if (title == 'My Outfits') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => const CreateOutfitScreen()),
            );
          } else if (title == 'Items') {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => const AddItemScreen()),
            );
          }
        },
        child: Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: AppColors.primaryWine,
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Icon(Icons.add, size: 18, color: AppColors.textLight),
        ),
      ),
    ],
  );
}

  Widget _buildWeatherOutfitCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.cardBg,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Row(
        children: [
          Expanded(
            flex: 5,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Today, Jan 1',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: AppColors.textDark),
                ),
                SizedBox(height: 8),
                Text('Weather', style: TextStyle(color: Colors.black54, fontSize: 12)),
                Text('-3° to +8°', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textDark)),
                SizedBox(height: 12),
                Text('Precipitation', style: TextStyle(color: Colors.black54, fontSize: 12)),
                Text('90%', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: AppColors.textDark)),
              ],
            ),
          ),
          Expanded(
            flex: 6,
            child: Container(
              height: 120,
              decoration: BoxDecoration(
                color: AppColors.cardBg,
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Center(
                child: Icon(Icons.checkroom, size: 48, color: AppColors.primaryWine),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildOutfitsList() {
    return SizedBox(
      height: 180,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: 3,
        separatorBuilder: (_, __) => const SizedBox(width: 16),
        itemBuilder: (context, index) {
          return Container(
            width: 150,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppColors.cardBg,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: AppColors.cardBg,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Center(
                      child: Icon(Icons.dry_cleaning, size: 40, color: AppColors.primaryWine),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  index % 2 == 0 ? 'Casual' : 'Classic',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: AppColors.textDark),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildItemsList() {
    return ListenableBuilder(
      listenable: WardrobeStore.instance,
      builder: (context, _) {
        final items = WardrobeStore.instance.items;
        if (items.isEmpty) return _buildEmptyItems();
        return SizedBox(
          height: 140,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: items.length,
            separatorBuilder: (_, __) => const SizedBox(width: 16),
            itemBuilder: (context, index) => _buildItemCard(items[index]),
          ),
        );
      },
    );
  }

  Widget _buildEmptyItems() {
    return Container(
      height: 140,
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.cardBg,
        borderRadius: BorderRadius.circular(18),
      ),
      child: const Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.checkroom, size: 32, color: AppColors.primaryWine),
          SizedBox(height: 8),
          Text(
            'No items yet',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: AppColors.textDark),
          ),
          SizedBox(height: 4),
          Text(
            'Tap + to add your first piece',
            style: TextStyle(fontSize: 12, color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }

  Widget _buildItemCard(WardrobeItem item) {
    return GestureDetector(
      onLongPress: () => _confirmDelete(item),
      child: Container(
        width: 120,
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: AppColors.cardBg,
          borderRadius: BorderRadius.circular(18),
        ),
        child: Column(
          children: [
            Expanded(
              child: WardrobeImage(
                bytes: item.imageBytes,
                url: item.imageUrl,
                fit: BoxFit.contain,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              item.displayName,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 12, color: AppColors.textDark),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _confirmDelete(WardrobeItem item) async {
    final remove = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.cardBg,
        title: const Text('Remove item?', style: TextStyle(color: AppColors.textDark)),
        content: Text(
          'Remove "${item.displayName}" from your wardrobe?',
          style: const TextStyle(color: AppColors.textDark),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel', style: TextStyle(color: AppColors.textMuted)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Remove', style: TextStyle(color: AppColors.primaryWine)),
          ),
        ],
      ),
    );
    if (remove == true) await WardrobeStore.instance.remove(item.id);
  }

  Widget _buildFloatingNavBar() {
    return Container(
      margin: const EdgeInsets.only(left: 24, right: 24, bottom: 20),
      height: 60,
      decoration: BoxDecoration(
        color: AppColors.primaryWine,
        borderRadius: BorderRadius.circular(32),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _navIcon(Icons.door_sliding_outlined, 0),
          _navIcon(Icons.auto_awesome_outlined, 1),
          _navIcon(Icons.grid_view_outlined, 2),
          _navIcon(Icons.person_outline, 3),
        ],
      ),
    );
  }

  Widget _navIcon(IconData icon, int index) {
    final bool isSelected = _currentIndex == index;
    return GestureDetector(
      onTap: () => setState(() => _currentIndex = index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: isSelected ? AppColors.background : Colors.transparent,
          shape: BoxShape.circle,
        ),
        child: Icon(
          icon,
          color: isSelected ? AppColors.primaryWine : AppColors.textLight,
        ),
      ),
    );
  }
}