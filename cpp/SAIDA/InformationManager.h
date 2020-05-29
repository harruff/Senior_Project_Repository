/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"

#include "AbstractManager.h"
#include "UnitData/UnitData.h"
#include "SaidaUtil.h"

#define INFO	InformationManager::Instance()
#define MYBASE INFO.getMainBaseLocation(S)->Center()
#define ENBASE INFO.getMainBaseLocation(E)->Center()

namespace MyBot
{
	enum TypeKind
	{
		AllUnitKind,
		AirUnitKind,
		GroundCombatKind, //���߿�..
		GroundUnitKind,
		BuildingKind,
		AllDefenseBuildingKind,
		AirDefenseBuildingKind,
		GroundDefenseBuildingKind,
		AllKind
	};

	/// ���� ��Ȳ���� �� �Ϻθ� ��ü �ڷᱸ�� �� �����鿡 �����ϰ� ������Ʈ�ϴ� class<br>
	/// ���� ���� ��Ȳ������ Broodwar �� ��ȸ�Ͽ� �ľ��� �� ������, ���� ���� ��Ȳ������ Broodwar �� ���� ��ȸ�� �Ұ����ϱ� ������ InformationManager���� ���� �����ϵ��� �մϴ�<br>
	/// ����, Broodwar �� BWEM ���� ���� ��ȸ�� �� �ִ� ���������� ��ó�� / ���� �����ϴ� ���� ������ �͵� InformationManager���� ���� �����ϵ��� �մϴ�
	class InformationManager : public AbstractManager
	{
		InformationManager() : AbstractManager("InformationManager") {};
		~InformationManager() {};

		/// �� �÷��̾�� (2�ο� ��, 3�ο� ��, 4�ο� ��, 8�ο� ��)
		int															mapPlayerLimit;

		/// Player - UnitData(�� Unit �� �� Unit�� UnitInfo �� Map ���·� �����ϴ� �ڷᱸ��) �� �����ϴ� �ڷᱸ�� ��ü<br>
		map<Player, UnitData>							_unitData;

		/// ��ü unit �� ������ ������Ʈ �մϴ� (UnitType, lastPosition, HitPoint ��. �����Ӵ� 1ȸ ����)
		void                    updateUnitsInfo();
		// enemy ���� ( Hide, Show )

		set<TechType>			researchedSet;
		map<UpgradeType, int>		upgradeSet;
		vector<UpgradeType>		upgradeList;

	protected:
		void updateManager();

	public:

		/// static singleton ��ü�� �����մϴ�
		static InformationManager &Instance();
		void initialize();

		Player       selfPlayer;		///< �Ʊ� Player
		Race			selfRace;		///< �Ʊ� Player�� ����
		Player       enemyPlayer;	///< ���� Player
		Race			enemyRace;		///< ���� Player�� ����
		Race		enemySelectRace;		///< ���� Player�� ������ ����

		/// Unit �� ���� ������ ������Ʈ�մϴ�
		void					onUnitShow(Unit unit);
		/// Unit �� ���� ������ ������Ʈ�մϴ�
		//		void					onUnitHide(Unit unit)        { updateUnitHide(unit, true); }
		/// Unit �� ���� ������ ������Ʈ�մϴ�
		void					onUnitCreate(Unit unit);
		/// Unit �� ���� ������ ������Ʈ�մϴ�
		void					onUnitComplete(Unit unit);
		/// ������ �ı�/����� ���, �ش� ���� ������ �����մϴ�
		void					onUnitDestroy(Unit unit);

		/// ���� ���� �ִ� �÷��̾�� (2�ο� ��, 3�ο� ��, 4�ο� ��, 8�ο� ��) �� �����մϴ�
		int						getMapPlayerLimit() {
			return mapPlayerLimit;
		}

		/// �ش� Player (�Ʊ� or ����) �� ��� ���� ��� UnitData �� �����մϴ�
		UnitData 				&getUnitData(Player player) {
			return _unitData[player];
		}

		// �ش� ������ UnitType �� ResourceDepot ����� �ϴ� UnitType�� �����մϴ�
		UnitType			getBasicResourceDepotBuildingType(Race race = Races::None);

		// �ش� ������ UnitType �� Refinery ����� �ϴ� UnitType�� �����մϴ�
		UnitType			getRefineryBuildingType(Race race = Races::None);

		// �ش� ������ UnitType �� SupplyProvider ����� �ϴ� UnitType�� �����մϴ�
		UnitType			getBasicSupplyProviderUnitType(Race race = Races::None);

		// �ش� ������ UnitType �� Worker �� �ش��ϴ� UnitType�� �����մϴ�
		UnitType			getWorkerType(Race race = Races::None);

		// �ش� ������ UnitType �� Basic Combat Unit �� �ش��ϴ� UnitType�� �����մϴ�
		UnitType			getBasicCombatUnitType(Race race = Races::None);

		// �ش� ������ UnitType �� Basic Combat Unit �� �����ϱ� ���� �Ǽ��ؾ��ϴ� UnitType�� �����մϴ�
		UnitType			getBasicCombatBuildingType(Race race = Races::None);

		// �ش� ������ UnitType �� Advanced Combat Unit �� �ش��ϴ� UnitType�� �����մϴ�
		UnitType			getAdvancedCombatUnitType(Race race = Races::None);

		// �ش� ������ UnitType �� Observer �� �ش��ϴ� UnitType�� �����մϴ�
		UnitType			getObserverUnitType(Race race = Races::None);

		// �ش� ������ UnitType �� Advanced Depense ����� �ϴ� UnitType�� �����մϴ�
		UnitType			getAdvancedDefenseBuildingType(Race race = Races::None, bool isAirDefense = false);


		// UnitData ���� API�� �Ʒ����� �����Ѵ�.
		UnitInfo *getUnitInfo(Unit unit, Player p);
		uList getUnits(UnitType t, Player p) {
			return _unitData[p].getUnitVector(t);
		}
		uList getBuildings(UnitType t, Player p) {
			return _unitData[p].getBuildingVector(t);
		}
		uMap &getUnits(Player p) {
			return _unitData[p].getAllUnits();
		}
		uMap &getBuildings(Player p) {
			return _unitData[p].getAllBuildings();
		}
		int			getCompletedCount(UnitType t, Player p);
		int			getDestroyedCount(UnitType t, Player p);
		int			getTotalCount(UnitType t, Player p) {
			return getAllCount(t, p) + getDestroyedCount(t, p);
		}
		map<UnitType, int> getDestroyedCountMap(Player p);
		int			getAllCount(UnitType t, Player p);

		void clearUnitNBuilding();

		uList getUnitsInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool ground = true, bool air = true, bool worker = true, bool hide = false, bool groundDist = false);
		uList getBuildingsInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool ground = true, bool air = true, bool hide = false, bool groundDist = false);
		uList getAllInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool ground = true, bool air = true, bool hide = false, bool groundDist = false);
		uList getUnitsInRectangle(Player p, Position leftTop, Position rightDown, bool ground = true, bool air = true, bool worker = true, bool hide = false);
		uList getBuildingsInRectangle(Player p, Position leftTop, Position rightDown, bool ground = true, bool air = true, bool hide = false);
		uList getAllInRectangle(Player p, Position leftTop, Position rightDown, bool ground = true, bool air = true, bool hide = false);
		uList getTypeUnitsInRadius(UnitType t, Player p, Position pos = Positions::Origin, int radius = 0, bool hide = false);
		uList getTypeBuildingsInRadius(UnitType t, Player p, Position pos = Positions::Origin, int radius = 0, bool incomplete = true, bool hide = true);
		uList getDefenceBuildingsInRadius(Player p, Position pos = Positions::Origin, int radius = 0, bool incomplete = true, bool hide = true);
		uList getTypeUnitsInRectangle(UnitType t, Player p, Position leftTop, Position rightDown, bool hide = false);
		uList getTypeBuildingsInRectangle(UnitType t, Player p, Position leftTop, Position rightDown, bool incomplete = true, bool hide = true);
		uList getUnitsInArea(Player p, Position pos, bool ground = true, bool air = true, bool worker = true, bool hide = true);
		uList getBuildingsInArea(Player p, Position pos, bool ground = true, bool air = true, bool hide = true);
		uList getTypeUnitsInArea(UnitType t, Player p, Position pos, bool hide = false);
		uList getTypeBuildingsInArea(UnitType t, Player p, Position pos, bool incomplete = true, bool hide = true);

		UnitInfo *getClosestUnit(Player p, Position pos, TypeKind kind = TypeKind::AllKind, int radius = 0, bool worker = false, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getFarthestUnit(Player p, Position pos, TypeKind kind = TypeKind::AllKind, int radius = 0, bool worker = false, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getClosestTypeUnit(Player p, Position pos, UnitType type, int radius = 0, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getClosestTypeUnit(Player p, Position pos, vector<UnitType> &types, int radius = 0, bool hide = false, bool groundDist = false, bool detectedOnly = true);
		UnitInfo *getFarthestTypeUnit(Player p, Position pos, UnitType type, int radius = 0, bool hide = false, bool groundDist = false, bool detectedOnly = true);

		bool hasResearched(TechType tech);
		void setResearched(UnitType unitType);
		void setUpgradeLevel();
		int getUpgradeLevel(UpgradeType up);

		vector<UnitType> getCombatTypes(Race race);
	};
}
