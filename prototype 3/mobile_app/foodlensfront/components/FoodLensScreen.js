import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  ScrollView,
  FlatList,
  BackHandler,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';

const { width, height } = Dimensions.get('window');

const FoodLensScreen = () => {
   const navigation = useNavigation();
  const [currentIndex, setCurrentIndex] = useState(0);
  const flatListRef = useRef(null);
  
  const carouselItems = [
    { id: '1', image: require('../assets/newafood.png') },
    { id: '2', image: require('../assets/newafood1.png') },
    { id: '3', image: require('../assets/newafood2.png') },
  ];

  // Auto-scroll functionality
  useEffect(() => {
    const interval = setInterval(() => {
      const nextIndex = (currentIndex + 1) % carouselItems.length;
      setCurrentIndex(nextIndex);
      flatListRef.current?.scrollToIndex({ index: nextIndex, animated: true });
    }, 3000); // Changed to 3 seconds for better user experience

    return () => clearInterval(interval);
  }, [currentIndex, carouselItems.length]);
 
  const handleDetail = () => {
    navigation.navigate('ImageUploading');
  }

   const handleFavourites = () => {
    navigation.navigate('FoodDetail');
  }

  const handleHome = () => {
    navigation.navigate('FoodLens');
  }
  

  const handleUpload = () => {
    navigation.navigate('ImageUploading');
  }
  
  
  const handlePerson= () => {
    navigation.navigate('Profile');
  }
  

  const onViewableItemsChanged = useRef(({ viewableItems }) => {
    if (viewableItems.length > 0) {
      setCurrentIndex(viewableItems[0].index);
    }
  }).current;

  const viewabilityConfig = useRef({
    itemVisiblePercentThreshold: 50,
  }).current;

  const renderCarouselItem = ({ item }) => (
    <View style={styles.carouselItem}>
      <Image source={item.image} style={styles.carouselImage} resizeMode="cover" />
    </View>
  );

  const goToSlide = (index) => {
    setCurrentIndex(index);
    flatListRef.current?.scrollToIndex({ index, animated: true });
  };

  const renderDots = () => (
    <View style={styles.dotsContainer}>
      {carouselItems.map((_, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.dot,
            { backgroundColor: index === currentIndex ? '#D94F4F' : '#DDD' }
          ]}
          onPress={() => goToSlide(index)}
        />
      ))}
    </View>
  );

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollContainer} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Image source={require('../assets/logo.png')} style={styles.logo} />
          </View>
          <TouchableOpacity>
            <Ionicons name="notifications-outline" size={24} color="#D94F4F" />
            <View style={styles.notificationBadge} />
          </TouchableOpacity>
        </View>

        {/* Image Slider */}
        <View style={styles.sliderContainer}>
          <FlatList
            ref={flatListRef}
            data={carouselItems}
            renderItem={renderCarouselItem}
            keyExtractor={(item) => item.id}
            horizontal
            pagingEnabled
            showsHorizontalScrollIndicator={false}
            onViewableItemsChanged={onViewableItemsChanged}
            viewabilityConfig={viewabilityConfig}
            contentContainerStyle={styles.flatListContent}
          />
          {renderDots()}
        </View>

        {/* Content */}
        <View style={styles.content}>
          <Text style={styles.title}>Food is Culture, Emotion & Memories</Text>
          <Text style={styles.description}>
            FoodLens is an innovative mobile app that blends AI technology with the rich heritage of Multi-cuisine. Simply upload a photo of any dish and FoodLens instantly detects its ingredients, generates authentic recipes, and helps you recreate traditional flavors at home.
          </Text>

          {/* Buttons */}
          <View style={styles.buttonContainer}>
            <TouchableOpacity onPress={handleFavourites}  style={styles.favoriteButton}>
              <Ionicons name="heart-outline" size={20} color="#D94F4F" />
              <Text style={styles.favoriteButtonText}>Favorites</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={handleDetail} style={styles.getStartedButton}>
              <Text style={styles.getStartedButtonText}>Get Started</Text>
              <Ionicons name="arrow-forward" size={20} color="#FFF" />
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>

      {/* Bottom Navigation - Now Static */}
      <View style={styles.bottomNav}>
        <TouchableOpacity onPress={handleHome} style={styles.navItem}>
          <Ionicons name="home" size={24} color="#D85A47" />
        </TouchableOpacity>
        <TouchableOpacity onPress={handleUpload}style={styles.navItem}>
          <Ionicons name="restaurant" size={24} color="#FFF" />
        </TouchableOpacity>
        
        <TouchableOpacity  onPress={handlePerson} style={styles.navItem}>
          <Ionicons name="person" size={24} color="#FFF" />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    paddingTop: 40,
  },
  scrollContainer: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  logo: {
    width: 70,
    height: 70,
    resizeMode: 'contain',
  },
  notificationBadge: {
    position: 'absolute',
    top: -5,
    right: -5,
    width: 10,
    height: 10,
    backgroundColor: '#D94F4F',
    borderRadius: 5,
  },
  sliderContainer: {
    height: 280,
    marginTop: 10,
    marginBottom: 30,
  },
  flatListContent: {
    alignItems: 'center',
  },
  carouselItem: {
    width: width * 0.8,
    height: 220,
    marginHorizontal: width * 0.1,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    backgroundColor: '#fff',
  },
  carouselImage: {
    width: '100%',
    height: '100%',
    borderRadius: 15,
  },
  dotsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 20,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginHorizontal: 5,
  },
  content: {
    padding: 20,
    alignItems: 'center',
    paddingBottom: 20, // Reduced padding since bottom nav is now static
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 10,
  },
  description: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 25,
  },
  buttonContainer: {
    flexDirection: 'row',
    marginTop: 20,
    justifyContent: 'space-between',
    width: '100%',
    paddingHorizontal: 20,
  },
  favoriteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#D94F4F',
    borderRadius: 25,
    paddingVertical: 10,
    paddingHorizontal: 20,
    backgroundColor: 'transparent',
  },
  favoriteButtonText: {
    color: '#D94F4F',
    fontSize: 16,
    marginLeft: 5,
  },
  getStartedButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#D94F4F',
    borderRadius: 25,
    paddingVertical: 10,
    paddingHorizontal: 20,
  },
  getStartedButtonText: {
    color: '#FFF',
    fontSize: 16,
    marginRight: 5,
  },
  bottomNav: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#333',
    paddingVertical: 15,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    // Removed position: 'absolute' and related properties
    // Added border top for better separation
    borderTopWidth: 1,
    borderTopColor: '#555',
  },
  navItem: {
    alignItems: 'center',
  },
});

export default FoodLensScreen;