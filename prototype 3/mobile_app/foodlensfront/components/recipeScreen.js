import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

const PizzaRecipeApp = () => {
  const [isFavorited, setIsFavorited] = useState(false);

  const ingredients = [
    'Flour',
    'Sausage',
    'Tomato',
    'Black pepper',
    'Salt',
    'Oregano',
    'Chili flakes'
  ];

  const recipeSteps = [
    'Make dough – Mix flour, yeast, water, salt, sugar, oil. Let rise 1 hr',
    'Make sauce – Cook tomatoes with garlic, salt, oregano.',
    'Assemble – Roll dough, spread sauce, add cheese & toppings.',
    'Bake – 475°F (245°C), 12–15 mins.',
    'Eat & enjoy!'
  ];

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
      
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Logo Section */}
        <View style={styles.logoContainer}>
          <Image
            source={require('../assets/logo.png')}
            style={styles.logo}
            resizeMode="contain"
          />
        </View>

        {/* Header Section */}
        <View style={styles.header}>
          <View style={styles.uploadStatus}>
            <View style={styles.statusIcon}>
              <Ionicons name="restaurant-outline" size={24} color="#C4342C" />
            </View>
            <Text style={styles.statusText}>Image Uploaded</Text>
            <View style={styles.checkIcon}>
              <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
            </View>
          </View>
        </View>

        {/* Pizza Image */}
        <View style={styles.imageContainer}>
          <Image
            source={
              require('../assets/pizza.jpg')
            }
            style={styles.pizzaImage}
            resizeMode="cover"
          />
        </View>

        {/* Upload Button */}
        <TouchableOpacity style={styles.uploadButton}>
          <Ionicons name="cloud-upload-outline" size={20} color="#C4342C" />
          <Text style={styles.uploadButtonText}>Upload your Image Here</Text>
        </TouchableOpacity>

        {/* View Recipe Button */}
        <TouchableOpacity style={styles.recipeButton}>
          <Text style={styles.recipeButtonText}>View Recipe & Ingredients</Text>
          <Ionicons name="arrow-forward" size={20} color="white" />
        </TouchableOpacity>

        {/* Title Section */}
        <View style={styles.titleSection}>
          <Text style={styles.title}>Pizza</Text>
          <TouchableOpacity
            onPress={() => setIsFavorited(!isFavorited)}
            style={styles.favoriteButton}
          >
            <Ionicons
              name={isFavorited ? "heart" : "heart-outline"}
              size={28}
              color={isFavorited ? "#C4342C" : "#666"}
            />
          </TouchableOpacity>
        </View>

        {/* Ingredients Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Ingredients</Text>
          {ingredients.map((ingredient, index) => (
            <Text key={index} style={styles.ingredientItem}>
              {ingredient}
            </Text>
          ))}
        </View>

        {/* Recipe Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recipe</Text>
          {recipeSteps.map((step, index) => (
            <View key={index} style={styles.recipeStep}>
              <Text style={styles.stepNumber}>{index + 1}.</Text>
              <Text style={styles.stepText}>{step}</Text>
            </View>
          ))}
        </View>

        {/* Bottom Spacing */}
        <View style={styles.bottomSpacing} />
      </ScrollView>

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="home-outline" size={24} color="#C4342C" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="restaurant-outline" size={24} color="#C4342C" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="bulb-outline" size={24} color="#666" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="person-outline" size={24} color="#666" />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  logoContainer: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
    alignItems: 'flex-start',
  },
  logo: {
    width: 80,
    height: 80,
  },
  header: {
    paddingHorizontal: 20,
    paddingTop: 10,
  },
  uploadStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  statusIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#C4342C',
  },
  statusText: {
    flex: 1,
    marginLeft: 15,
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  checkIcon: {
    width: 30,
    height: 30,
  },
  imageContainer: {
    margin: 20,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  pizzaImage: {
    width: '100%',
    height: 200,
  },
  uploadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginBottom: 15,
    paddingVertical: 12,
    borderWidth: 2,
    borderColor: '#C4342C',
    borderRadius: 8,
    backgroundColor: 'white',
  },
  uploadButtonText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#C4342C',
    fontWeight: '500',
  },
  recipeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginBottom: 20,
    paddingVertical: 15,
    backgroundColor: '#C4342C',
    borderRadius: 8,
  },
  recipeButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginRight: 8,
  },
  titleSection: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 20,
    marginBottom: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#C4342C',
  },
  favoriteButton: {
    padding: 5,
  },
  section: {
    marginHorizontal: 20,
    marginBottom: 25,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  ingredientItem: {
    fontSize: 16,
    color: '#333',
    marginBottom: 8,
    paddingLeft: 5,
  },
  recipeStep: {
    flexDirection: 'row',
    marginBottom: 12,
    alignItems: 'flex-start',
  },
  stepNumber: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginRight: 8,
  },
  stepText: {
    flex: 1,
    fontSize: 16,
    color: '#333',
    lineHeight: 22,
  },
  bottomSpacing: {
    height: 20,
  },
  bottomNav: {
    flexDirection: 'row',
    backgroundColor: '#2C2C2C',
    paddingVertical: 15,
    paddingHorizontal: 20,
    justifyContent: 'space-around',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  navItem: {
    padding: 5,
  },
});

export default PizzaRecipeApp;