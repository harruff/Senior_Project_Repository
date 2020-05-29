/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"
#include "InformationManager.h"
#include "UXManager.h"

namespace MyBot
{
	/// ���� �����α׷��� ��ü�� �Ǵ� class<br>
	/// ��Ÿũ����Ʈ ��� ���� �߻��ϴ� �̺�Ʈ���� �����ϰ� ó���ǵ��� �ش� Manager ��ü���� �̺�Ʈ�� �����ϴ� ������ Controller ������ �մϴ�
	class GameCommander
	{
	public:

		static GameCommander &Instance();

		GameCommander();
		~GameCommander();

		void update();

		/// ��Ⱑ ���۵� �� ��ȸ������ �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onStart();
		/// ��Ⱑ ����� �� ��ȸ������ �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onEnd(bool isWinner);

		/// ����(�ǹ�/��������/��������)�� Create �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onUnitCreate(Unit unit);
		///  ����(�ǹ�/��������/��������)�� Destroy �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onUnitDestroy(Unit unit);

		/// ����(�ǹ�/��������/��������)�� Morph �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�<br>
		/// Zerg ������ ������ �ǹ� �Ǽ��̳� ��������/�������� ���꿡�� ���� ��κ� Morph ���·� ����˴ϴ�
		void onUnitMorph(Unit unit);

		/// ����(�ǹ�/��������/��������)�� �Ҽ� �÷��̾ �ٲ� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�<br>
		/// Gas Geyser�� � �÷��̾ Refinery �ǹ��� �Ǽ����� ��, Refinery �ǹ��� �ı��Ǿ��� ��, Protoss ���� Dark Archon �� Mind Control �� ���� �Ҽ� �÷��̾ �ٲ� �� �߻��մϴ�
		void onUnitRenegade(Unit unit);
		/// ����(�ǹ�/��������/��������)�� �ϴ� �� (�ǹ� �Ǽ�, ���׷��̵�, �������� �Ʒ� ��)�� ������ �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onUnitComplete(Unit unit);

		/// ����(�ǹ�/��������/��������)�� Discover �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�<br>
		/// �Ʊ� ������ Create �Ǿ��� �� ��簡, ���� ������ Discover �Ǿ��� �� �߻��մϴ�
		void onUnitDiscover(Unit unit);
		/// ����(�ǹ�/��������/��������)�� Evade �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�<br>
		/// ������ Destroy �� �� �߻��մϴ�
		void onUnitEvade(Unit unit);

		/// ����(�ǹ�/��������/��������)�� Show �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�<br>
		/// �Ʊ� ������ Create �Ǿ��� �� ��簡, ���� ������ Discover �Ǿ��� �� �߻��մϴ�
		void onUnitShow(Unit unit);
		/// ����(�ǹ�/��������/��������)�� Hide �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�<br>
		/// ���̴� ������ Hide �� �� �߻��մϴ�
		void onUnitHide(Unit unit);

		/// Unit�� Landing �Ǿ��� ��
		void onUnitLanded(Unit unit);
		/// Unit�� Lift �Ǿ��� ��
		void onUnitLifted(Unit unit);

		// BasicBot 1.1 Patch Start ////////////////////////////////////////////////
		// onNukeDetect, onPlayerLeft, onSaveGame �̺�Ʈ�� ó���� �� �ֵ��� �޼ҵ� �߰�

		/// �ٹ̻��� �߻簡 �����Ǿ��� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onNukeDetect(Position target);

		/// �ٸ� �÷��̾ ����� ������ �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onPlayerLeft(Player player);

		/// ������ ������ �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onSaveGame(string gameName);

		// BasicBot 1.1 Patch End //////////////////////////////////////////////////

		/// �ؽ�Ʈ�� �Է� �� ���͸� �Ͽ� �ٸ� �÷��̾�鿡�� �ؽ�Ʈ�� �����Ϸ� �� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onSendText(string text);
		/// �ٸ� �÷��̾�κ��� �ؽ�Ʈ�� ���޹޾��� �� �߻��ϴ� �̺�Ʈ�� ó���մϴ�
		void onReceiveText(Player player, string text);
	};

}