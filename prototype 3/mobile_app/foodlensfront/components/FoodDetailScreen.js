import React from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  SafeAreaView,
  StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons'; 
import { useNavigation } from '@react-navigation/native';

const FoodDetailScreen = () => {
  // Sample data for the list with local images
  const foodItems = [
    {
      id: '1',
      title: 'SAMAY BAJI',
      description:
        'Samay Baji is a traditional Newari dish from the Kathmandu Valley in Nepal. It is typically served during festivals or any events.',
      image: require('../assets/samay_baji_2.png'), // Path to your local image
    },
    {
      id: '2',
      title: 'ANTIPASTO PLATTER',
      description:
        'An Italian appetizer spread with marinated vegetables, mozzarella, prosciutto, olives, artichoke hearts, and crusty bread. This platter combines bold, zesty flavors with a rustic charm, perfect for starting a meal with a Mediterranean flair.',
      image: require('../assets/food_platter.png'), // Path to your local image
    },
    {
      id: '3', // Fixed duplicate ID to avoid FlatList key conflict
      title: 'SAMAY BAJI',
      description:
        'Samay Baji is a traditional Newari dish from the Kathmandu Valley in Nepal. It is typically served during festivals or any events.',
      image: require('../assets/samay_baji_2.png'), // Path to your local image
    },
  ];
     const navigation = useNavigation();
const handleHome = () => {
    navigation.navigate('FoodLens');
  }

  const handlePerson = () => {
    navigation.navigate('Profile');
  }
  // Render each food item card
  const renderFoodItem = ({ item }) => (
    <View style={styles.card}>
      <Image source={item.image} style={styles.cardImage} />
      <View style={styles.cardContent}>
        <View style={styles.titleContainer}>
          <Ionicons name="heart-outline" size={20} color="#FF4D4D" style={styles.heartIcon} />
          <Text style={styles.title}>{item.title}</Text>
        </View>
Spinner        <Text style={styles.description}>{item.description}</Text>
        <TouchableOpacity style={styles.button}>
          <Text style={styles.buttonText}>View Details</Text>
          <Ionicons name="flame" size={16} color="#FF4D4D" style={styles.flameIcon} />
        </TouchableOpacity>
      </View>
    </View>
  );

  // Header to be rendered as the first item in FlatList
  const renderHeader = () => (
    <View style={styles.header}>
      <Image source={require('../assets/logo.png')} style={styles.logoIcon} />
      <View style={styles.notificationContainer}>
        <Ionicons name="notifications-outline" size={24} color="#FF4D4D" />
        <View style={styles.badge}>
          <Text style={styles.badgeText}>1</Text>
        </View>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor="#ffffff" />
      {/* Main Content */}
      <FlatList
        data={foodItems}
        renderItem={renderFoodItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContainer}
        ListHeaderComponent={renderHeader} // Add the header as part of the FlatList
      />

      {/* Footer */}
      {/* <View style={styles.footer}>
        <Ionicons name="heart" size={20} color="#FF4D4D" />
        <Text style={styles.footerText}>Food is LOVE</Text>
        <Ionicons name="heart" size={20} color="#FF4D4D" />
      </View> */}
    <View style={styles.bottomNav}>
        <TouchableOpacity onPress={handleHome} style={styles.navItem}>
          <Ionicons name="home" size={24} color="#FFF" />
        </TouchableOpacity>
        <TouchableOpacity style={[styles.navItem, styles.activeNavItem]}>
          <Ionicons name="restaurant" size={24} color="#D85A47" />
        </TouchableOpacity>
        {/* <TouchableOpacity style={styles.navItem}>
          <Ionicons name="search" size={24} color="#999" />
        </TouchableOpacity> */}
        <TouchableOpacity onPress={handlePerson} style={styles.navItem}>
          <Ionicons name="person" size={24} color="#FFF" />
        </TouchableOpacity>
      </View>

    </SafeAreaView>
  );
};

// Styles
const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#ffffff', // Changed to match the main content background
    paddingTop: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 10,
    backgroundColor: '#ffffff', // Same as listContainer background for seamless integration
  },
  logoIcon: {
    height: 80,
    width: 80,
  },
  notificationContainer: {
    position: 'relative',
  },
  badge: {
    position: 'absolute',
    right: -6,
    top: -3,
    backgroundColor: '#FF4D4D',
    borderRadius: 6,
    width: 12,
    height: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  badgeText: {
    color: '#FFF',
    fontSize: 8,
    fontWeight: 'bold',
  },
  listContainer: {
    padding: 15,
    backgroundColor: '#ffffff',
    paddingBottom: 60,
  },
  card: {
    flexDirection: 'row',
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 3,
  },
  cardImage: {
    width: 100,
    height: 100,
    borderRadius: 10,
    marginRight: 15,
  },
  cardContent: {
    flex: 1,
    justifyContent: 'space-between',
  },
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  heartIcon: {
    marginRight: 5,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#000',
  },
  description: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 10,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 15,
  },
  buttonText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#000',
    marginRight: 5,
  },
  flameIcon: {
    marginLeft: 5,
  },
  footer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    paddingVertical: 10,
  },
  footerText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: 'bold',
    marginHorizontal: 5,
  },
  bottomNav: {
    flexDirection: 'row',
    backgroundColor: '#2A2A2A',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  navItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 10,
  },
  activeNavItem: {
    backgroundColor: 'rgba(216, 90, 71, 0.15)',
    borderRadius: 8,
  },
});

export default FoodDetailScreen;