import 'package:flutter/material.dart';
import '../../../core/constants/app_colors.dart';

class CreateOutfitScreen extends StatefulWidget {
  const CreateOutfitScreen({super.key});

  @override
  State<CreateOutfitScreen> createState() => _CreateOutfitScreenState();
}

class _CreateOutfitScreenState extends State<CreateOutfitScreen> {
  String selectedWeather = 'Cold';
  String selectedCategory = 'Casual';

  final List<String> weatherOptions = ['Extra cold', 'Very cold', 'Cold', 'Chilly', 'Warm', 'Hot'];
  final List<String> categories = ['Classic', 'Sport', 'Casual', 'Festive', 'Home', 'Outside'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Create Outfit'),
        actions: [
          IconButton(
            icon: const Icon(Icons.check_circle_outline, size: 26),
            onPressed: () {},
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              height: 220,
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppColors.cardBg,
                borderRadius: BorderRadius.circular(24),
              ),
              child: const Center(
                child: Icon(Icons.checkroom, size: 90, color: AppColors.primaryWine),
              ),
            ),
            const SizedBox(height: 24),

            const Text('Select Items', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textDark)),
            const SizedBox(height: 12),
            SizedBox(
              height: 120,
              child: ListView(
                scrollDirection: Axis.horizontal,
                children: [
                  _buildSelectedItemCard('Brown Jacket'),
                  const SizedBox(width: 12),
                  _buildSelectedItemCard('Camel Shirt'),
                ],
              ),
            ),
            const SizedBox(height: 24),

            const Text('Select Weather', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textDark)),
            const SizedBox(height: 12),
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 2.5,
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
              ),
              itemCount: weatherOptions.length,
              itemBuilder: (context, index) {
                final weather = weatherOptions[index];
                final isSelected = selectedWeather == weather;
                return GestureDetector(
                  onTap: () => setState(() => selectedWeather = weather),
                  child: Container(
                    decoration: BoxDecoration(
                      color: isSelected ? AppColors.primaryWine : AppColors.cardBg,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      weather,
                      style: TextStyle(
                        color: isSelected ? AppColors.textLight : AppColors.textDark,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSelectedItemCard(String title) {
    return Container(
      width: 110,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: AppColors.cardBg,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: AppColors.cardBg,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(child: Icon(Icons.dry_cleaning, size: 32, color: AppColors.primaryWine)),
            ),
          ),
          const SizedBox(height: 6),
          Text(title, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: AppColors.textDark)),
        ],
      ),
    );
  }
}