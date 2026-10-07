import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../core/constants/app_colors.dart';
import 'dashboard_screen.dart';

class LandingPage extends StatelessWidget {
  const LandingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      
      // Side Navigation Drawer
      drawer: Drawer(
        backgroundColor: AppColors.background,
        child: ListView(
          padding: const EdgeInsets.symmetric(vertical: 50, horizontal: 20),
          children: [
            Text(
              'Vestara',
              style: GoogleFonts.meaCulpa(
                fontSize: 48,
                fontWeight: FontWeight.bold,
                color: AppColors.primaryWine,
              ),
            ),
            const SizedBox(height: 10),
            const Text(
              'Personal AI Wardrobe',
              style: TextStyle(color: AppColors.textMuted, fontSize: 12),
            ),
            const Divider(height: 40, color: AppColors.cardBorder),
            _buildDrawerTile(Icons.door_sliding_outlined, 'My Closet', () {}),
            _buildDrawerTile(Icons.auto_awesome_outlined, 'AI Stylist Agent', () {}),
            _buildDrawerTile(Icons.calendar_today_outlined, 'Weekly Planner', () {}),
            _buildDrawerTile(Icons.analytics_outlined, 'Style Analytics', () {}),
            _buildDrawerTile(Icons.settings_outlined, 'Settings', () {}),
          ],
        ),
      ),

      body: SafeArea(
        child: Builder(
          builder: (context) => SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 12.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Custom Top Header
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // Hamburger Menu Button
                    IconButton(
                      icon: const Icon(Icons.menu, color: AppColors.textDark, size: 28),
                      onPressed: () => Scaffold.of(context).openDrawer(),
                    ),

                    // Vestara Name with GoogleFonts.meaCulpa
                    Text(
  'Vestara',
  style: TextStyle(
    fontFamily: 'MeaCulpa',
    fontSize: 48,
    color: AppColors.primaryWine,
  ),
),

                    // Notification Icon with Red Badge
                    Stack(
                      children: [
                        IconButton(
                          icon: const Icon(Icons.notifications_none_outlined, color: AppColors.textDark, size: 26),
                          onPressed: () {},
                        ),
                        Positioned(
                          right: 10,
                          top: 10,
                          child: Container(
                            width: 8,
                            height: 8,
                            decoration: const BoxDecoration(
                              color: AppColors.primaryWine,
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // Greeting & Subtitle
                const Text(
                  'Good Morning, Aastha',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textDark,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Here is your AI-curated outfit for today based on 22°C Clear Sky.',
                  style: TextStyle(fontSize: 13, color: AppColors.textMuted),
                ),
                const SizedBox(height: 20),

                // Centerpiece Card: Today's Recommendation
                GestureDetector(
                  onTap: () {
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(builder: (context) => const DashboardScreen()),
                    );
                  },
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: AppColors.cardBg,
                      borderRadius: BorderRadius.circular(28),
                      border: Border.all(color: AppColors.cardBorder, width: 1.5),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.04),
                          blurRadius: 15,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        // Recommendation Tag
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                              decoration: BoxDecoration(
                                color: AppColors.primaryWine.withOpacity(0.12),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: const Text(
                                'TODAY\'S RECOMMENDATION',
                                style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                  color: AppColors.primaryWine,
                                  letterSpacing: 1.0,
                                ),
                              ),
                            ),
                            const Text(
                              '96% Match',
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                                color: AppColors.textDark,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 20),

                        // Layered Clothes Visual Display
                        Container(
                          height: 220,
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: AppColors.background,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Stack(
                            alignment: Alignment.center,
                            children: const [
                              Icon(Icons.checkroom, size: 90, color: AppColors.primaryWine),
                              Positioned(
                                bottom: 12,
                                child: Text(
                                  'Beige Trench & White Denim',
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.bold,
                                    color: AppColors.textDark,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 20),

                        // Card Action CTA
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const [
                            Text(
                              'Tap to Explore Your Wardrobe',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.bold,
                                color: AppColors.primaryWine,
                              ),
                            ),
                            SizedBox(width: 8),
                            Icon(Icons.arrow_forward_ios, size: 14, color: AppColors.primaryWine),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Quick Closet Overview Stats
                Row(
                  children: [
                    _buildStatCard('42', 'Items Digitized'),
                    const SizedBox(width: 12),
                    _buildStatCard('18', 'Looks Created'),
                    const SizedBox(width: 12),
                    _buildStatCard('92%', 'Style Accuracy'),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildDrawerTile(IconData icon, String title, VoidCallback onTap) {
    return ListTile(
      leading: Icon(icon, color: AppColors.primaryWine),
      title: Text(
        title,
        style: const TextStyle(color: AppColors.textDark, fontWeight: FontWeight.bold),
      ),
      onTap: onTap,
    );
  }

  Widget _buildStatCard(String value, String label) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 10),
        decoration: BoxDecoration(
          color: AppColors.cardBg,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppColors.cardBorder.withOpacity(0.6)),
        ),
        child: Column(
          children: [
            Text(
              value,
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: AppColors.primaryWine),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 10, color: AppColors.textMuted, fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }
}